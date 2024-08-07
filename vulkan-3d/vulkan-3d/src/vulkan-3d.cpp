#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

//#define PROFILE_MAIN_LOOP
//#define PROFILE_COMMAND_BUFFERS
#define ENABLE_DEBUG

#include "../ext/tiny_gltf.h" // load .obj and .mtl files
#include "../ext/stb_image_resize.h"

// headers
#include "dml.hpp"
#include "dvl.hpp"
#include "utils.hpp"
#include "vkhelper.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include "../ext/taskflow/taskflow.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <optional> //allows to create optional values in a more efficient way than using pointers
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream> //allows to read and write files
#include <array>
#include <chrono> //time library
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <random>
#include <ctime> //random seed based on time
#include <cmath>
#include <memory>

using microseconds = std::chrono::microseconds;
using milliseconds = std::chrono::milliseconds;

constexpr uint32_t MAX_MODELS = 1200;
constexpr uint32_t MAX_LIGHTS = 300;

constexpr uint32_t SCREEN_WIDTH = 3200;
constexpr uint32_t SCREEN_HEIGHT = 1800;

const std::string SHADER_DIR = "shaders/compiled/";
const std::string MODEL_DIR = "assets/models/";
const std::string SKYBOX_DIR = "assets/skyboxes/";
const std::string FONT_DIR = "assets/fonts/";

bool rtSupported = false; // a bool if raytracing is supported on the device

struct CamData {
	dml::vec3 camPos; //x, y, z
	dml::vec4 quat;
	float upAngle;
	float rightAngle;

	dml::mat4 projectionMatrix;
	dml::mat4 viewMatrix;

	// buffers for the camera matrix ubo
	VkBuffer buffer;
	VkDeviceMemory bufferMem;

	float lastX;
	float lastY;
	bool locked;

	float fov;
	float nearP;
	float farP;

	CamData()
		: camPos(0.0f, 0.0f, 0.0f),
		quat(0.0f, 0.0f, 0.0f, 1.0f),
		upAngle(0.0f),
		rightAngle(0.0f),
		projectionMatrix(),
		viewMatrix(),
		buffer(VK_NULL_HANDLE),
		bufferMem(VK_NULL_HANDLE),
		lastX(0.0f),
		lastY(0.0f),
		locked(true),
		fov(60.0f),
		nearP(0.1f),
		farP(1000.0f)
	{}

	dml::mat4 getViewMatrix() {
		return dml::viewMatrix(camPos, dml::radians(upAngle), dml::radians(rightAngle));
	}

	void updateQuaternion() {
		dml::vec4 yRot = dml::angleAxis(dml::radians(upAngle), dml::vec3(1, 0, 0));
		dml::vec4 xRot = dml::angleAxis(dml::radians(rightAngle), dml::vec3(0, 1, 0));
		quat = yRot * xRot;
	}
};

// globals
CamData cam;
VkDevice device;
VkQueue graphicsQueue;
VkPhysicalDevice physicalDevice;
GLFWwindow* window = nullptr;

class Engine {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
	}
private:
	struct Light { // spotlight
		dml::vec3 pos;
		dml::vec3 col;
		dml::vec3 target;
		float baseIntensity;
		dml::mat4 proj;
		dml::mat4 view;
		float innerConeAngle; // in degrees
		float outerConeAngle; // in degrees
		float constantAttenuation;
		float linearAttenuation;
		float quadraticAttenuation;
		dvl::Texture shadowMapData;
		VkFramebuffer frameBuffer;
		bool followPlayer;


		// default constructor
		Light()
			: pos(dml::vec3(0.0f, 0.0f, 0.0f)),
			col(dml::vec3(1.0f, 1.0f, 1.0f)),
			target(dml::vec3(0.0f, 0.0f, 0.0f)),
			baseIntensity(1.0f),
			proj(dml::mat4(1.0f)),
			view(dml::mat4(1.0f)),
			innerConeAngle(6.6f),
			outerConeAngle(10.0f),
			constantAttenuation(1.0f),
			linearAttenuation(0.1f),
			quadraticAttenuation(0.032f),
			shadowMapData(),
			frameBuffer(VK_NULL_HANDLE),
			followPlayer(false) {
		}

		// copy constructor
		Light(const Light& other)
			: pos(other.pos),
			col(other.col),
			target(other.target),
			baseIntensity(other.baseIntensity),
			proj(other.proj),
			view(other.view),
			innerConeAngle(other.innerConeAngle),
			outerConeAngle(other.outerConeAngle),
			constantAttenuation(other.constantAttenuation),
			linearAttenuation(other.linearAttenuation),
			quadraticAttenuation(other.quadraticAttenuation),
			shadowMapData(other.shadowMapData),
			frameBuffer(other.frameBuffer),
			followPlayer(other.followPlayer) {
		}
	};
	struct LightMatrix {
		dml::mat4 view;
		dml::mat4 proj;
	};

	struct LightDataObject {
		dml::mat4 view;
		dml::mat4 proj;

		dml::vec3 pos;
		dml::vec3 col;
		dml::vec3 target;
		float baseIntensity;
		float innerConeAngle; // in degrees
		float outerConeAngle; // in degrees
		float constantAttenuation;
		float linearAttenuation;
		float quadraticAttenuation;

		LightDataObject() {
			memset(this, 0, sizeof(LightDataObject)); //memset everything to 0
		}
	};

	struct LightDataSSBO {
		std::vector<LightDataObject> lightCords;
		size_t memSize;
	};

	struct ModelMat {
		dml::mat4 model;
		uint32_t render;

		ModelMat()
			: model(),
			render(true)
		{}

	};
	struct ModelMatInstanceData {
		ModelMat object[MAX_MODELS];
	};
	struct CamUBO {
		dml::mat4 view;
		dml::mat4 proj;
	};

	struct ShadowMapDim {
		uint32_t mapWidth = 2048;
		uint32_t mapHeight = 2048;
	};

	struct BufData {
		uint32_t vertexOffset;
		uint32_t vertexCount;
		uint32_t indexOffset;
		uint32_t indexCount;
	};

	struct SkyboxObject { // skybox struct
		dvl::Texture cubemap;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		BufData bufferData; // buffer data for the skybox (vert offsets, etc)
		VkBuffer vertBuffer;
		VkDeviceMemory vertBufferMem;
		VkBuffer indBuffer;
		VkDeviceMemory indBufferMem;

		std::vector<dml::vec3> vertices;
		std::vector<uint32_t> indices;

		SkyboxObject()
			: cubemap(),
			pipelineLayout(VK_NULL_HANDLE),
			pipeline(VK_NULL_HANDLE),
			bufferData(),
			vertBuffer(VK_NULL_HANDLE),
			vertBufferMem(VK_NULL_HANDLE),
			indBuffer(VK_NULL_HANDLE),
			indBufferMem(VK_NULL_HANDLE),

			indices{
				0, 1, 2, 2, 3, 0,
				7, 6, 5, 5, 4, 7,
				4, 5, 1, 1, 0, 4,
				3, 2, 6, 6, 7, 3,
				4, 0, 3, 3, 7, 4,
				1, 5, 6, 6, 2, 1
			},
			vertices{
				{-1.0f,  1.0f,  1.0f},
				{-1.0f, -1.0f,  1.0f},
				{ 1.0f, -1.0f,  1.0f},
				{ 1.0f,  1.0f,  1.0f},
				{-1.0f,  1.0f, -1.0f},
				{-1.0f, -1.0f, -1.0f},
				{ 1.0f, -1.0f, -1.0f},
				{ 1.0f,  1.0f, -1.0f}
			}
		{}
	};
	struct DSObject {
		std::vector<VkDescriptorSetLayout> layouts;
		std::vector<VkDescriptorSet> sets;
		std::vector<VkDescriptorPool> pools;
	};

	struct PipelineData {
		VkRenderPass renderPass;
		VkPipelineLayout layout;
		VkPipeline graphicsPipeline;
	};

	struct SCData {
		VkSwapchainKHR swapChain;
		std::vector<VkImage> images;
		VkFormat imageFormat;
		VkExtent2D extent;
		std::vector<VkImageView> imageViews;
		uint32_t imageCount;
		std::vector<VkFramebuffer> framebuffers;

		SCData()
			: swapChain(VK_NULL_HANDLE),
			images(),
			imageFormat(VK_FORMAT_UNDEFINED),
			extent(),
			imageViews(),
			imageCount(0),
			framebuffers()
		{}
	};

	struct KeyPO {
		bool pressedLastFrame = false;
		int keyPress;

		KeyPO() : pressedLastFrame(false), keyPress(-1) {}
		KeyPO(const int key) : pressedLastFrame(false), keyPress(key) {}

		bool isPressed() {
			bool notPressedLast = !pressedLastFrame;
			pressedLastFrame = (glfwGetKey(window, keyPress) == GLFW_PRESS);
			return (glfwGetKey(window, keyPress) == GLFW_PRESS) && notPressedLast;
		}
	};

	struct WBOITData { // weighted blended order independent transparency
		dvl::Texture weightedColor;
		VkFramebuffer frameBuffer;

		WBOITData()
			: weightedColor(),
			frameBuffer(VK_NULL_HANDLE)
		{}
	};

	struct MainPassTex {
		dvl::Texture depth;
		dvl::Texture color;
	};

	struct CommandBufferCollection {
		std::vector<VkCommandBuffer> buffers;
		std::vector<VkCommandPool> pools;

		void resize(size_t size) {
			buffers.resize(size);
			pools.resize(size);
		}

		size_t size() const {
			return buffers.size();
		}

		VkCommandBuffer* data() {
			return buffers.data();
		}

		VkCommandBuffer& operator[](size_t index) {
			return buffers[index];
		}
	};

	struct CommandBufferSet {
		CommandBufferCollection primary;
		CommandBufferCollection secondary;
	};

	// window and rendering context
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	VkInstance instance = VK_NULL_HANDLE;
	VkQueue presentQueue = VK_NULL_HANDLE;
	VkQueue computeQueue = VK_NULL_HANDLE;
	VkQueue transferQueue = VK_NULL_HANDLE;
	vkhelper::QueueFamilyIndices queueFamilyIndices;

	// key press objects
	KeyPO escapeKey = KeyPO(GLFW_KEY_ESCAPE);
	KeyPO eKey = KeyPO(GLFW_KEY_E);
	KeyPO rKey = KeyPO(GLFW_KEY_R);

	// swap chain and framebuffers
	SCData swap = {};
	size_t currentFrame = 0;

	// viewport config
	VkViewport swapVP{};

	// rendering pipeline data
	PipelineData mainPassPipeline;
	PipelineData shadowMapPipeline;
	PipelineData compPipelineData;
	PipelineData wboitPipeline;

	VkFramebuffer mainPassFB = VK_NULL_HANDLE;
	MainPassTex mainPassTextures = {};

	// command buffers and command pool
	VkCommandPool commandPool = VK_NULL_HANDLE;
	CommandBufferSet mainPassCommandBuffers;
	CommandBufferSet shadowMapCommandBuffers;
	CommandBufferSet wboitCommandBuffers;
	CommandBufferSet compCommandBuffers;

	// buffers
	VkBuffer vertBuffer = VK_NULL_HANDLE;
	VkBuffer indBuffer = VK_NULL_HANDLE;
	VkBuffer instanceBuffer = VK_NULL_HANDLE;
	VkBuffer lightBuffer = VK_NULL_HANDLE;
	VkBuffer sceneIndexBuffer = VK_NULL_HANDLE;

	// buffer memory
	VkDeviceMemory vertBufferMem = VK_NULL_HANDLE;
	VkDeviceMemory indBufferMem = VK_NULL_HANDLE;
	VkDeviceMemory instanceBufferMem = VK_NULL_HANDLE;
	VkDeviceMemory lightBufferMem = VK_NULL_HANDLE;
	VkDeviceMemory sceneIndexBufferMem = VK_NULL_HANDLE;

	// synchronization primitives
	std::vector<VkFence> inFlightFences;
	VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
	VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
	VkSemaphore shadowSemaphore = VK_NULL_HANDLE;
	VkSemaphore wboitSemaphore = VK_NULL_HANDLE;
	VkSemaphore compSemaphore = VK_NULL_HANDLE;

	// shader modules
	VkShaderModule fragShaderModule = VK_NULL_HANDLE;
	VkShaderModule vertShaderModule = VK_NULL_HANDLE;

	// depth buffering
	VkFormat depthFormat = VK_FORMAT_UNDEFINED;
	WBOITData wboit = {};

	// descriptor sets and pools
	DSObject descs = {};
	VkDescriptorSetLayout imguiDescriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool imguiDescriptorPool = VK_NULL_HANDLE;
	PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = nullptr;

	// scene data and objects
	std::vector<BufData> bufferData;
	std::vector<std::unique_ptr<dvl::Mesh>> objects;
	std::vector<std::unique_ptr<dvl::Mesh>> originalObjects;
	std::vector<uint32_t> playerModels;
	ModelMatInstanceData objInstanceData = {};
	CamUBO camMatData = {};
	LightDataSSBO lightData = {};
	std::vector<std::unique_ptr<Light>> lights;

	ShadowMapDim shadowProps = {};
	uint32_t modelIndex = 0; // index of where vertecies are loaded to

	std::unordered_map<size_t, size_t> uniqueModelIndex;
	std::unordered_map<size_t, size_t> modelHashToBufferIndex;

	// textures and materials
	std::vector<dvl::Texture> allTextures;
	std::vector<VkDescriptorImageInfo> shadowInfos;
	std::vector<int> meshTexStartInd;
	size_t totalTextureCount = 0;
	unsigned char* imageData = nullptr;

	// skybox data
	float* skyboxData = nullptr;
	SkyboxObject skybox = {};

	// font data
	ImFont* font_small = nullptr;
	ImFont* font_large = nullptr;

	// performance metrics
	uint32_t fps = 0;
	double lastFrame = 0.0;

	// mutexes and multithreading
	std::mutex modelMtx;
	std::mutex compositionMutex;

	tf::Executor cmdExecutor;
	tf::Taskflow tfCmd;

	tf::Executor meshExecutor;
	tf::Taskflow tfMesh;

	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // enable window resizing

		window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Vulkan", nullptr, nullptr);
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForVulkan(window, true);

		font_large = ImGui::GetIO().Fonts->AddFontFromFileTTF((FONT_DIR + "OpenSans/OpenSans-VariableFont_wdth,wght.ttf").c_str(), 50.0f);
	}

	void createObject(std::string path, dml::vec3 scale, dml::vec4 rotation, dml::vec3 pos) {
		loadModel(scale, pos, rotation, MODEL_DIR + path);
	}

	Light createLight(dml::vec3 pos, dml::vec3 t, dml::vec3 color = { 1.0f, 1.0f, 1.0f }, float intensity = 0.6f) {
		Light l;
		l.col = color;
		l.pos = pos;
		l.baseIntensity = intensity;
		l.target = t;
		l.constantAttenuation = 1.0f;
		l.linearAttenuation = 0.1f;
		l.quadraticAttenuation = 0.032f;
		l.innerConeAngle = 6.6f;
		l.outerConeAngle = 10.0f;
		l.followPlayer = false;
		return l;
	}

	Light createPlayerLight(dml::vec3 color = { 1.0f, 1.0f, 1.0f }, float intensity = 0.6f) {
		Light l;
		l.col = color;
		l.baseIntensity = intensity;
		l.constantAttenuation = 1.0f;
		l.linearAttenuation = 0.1f;
		l.quadraticAttenuation = 0.032f;
		l.innerConeAngle = 6.6f;
		l.outerConeAngle = 20.0f;
		l.followPlayer = true;
		return l;
	}

	void setPlayer(uint16_t i) {
		auto p = std::make_unique<dvl::Mesh>(*objects[i]);
		p->player = true;
		p->scale = dml::vec3(0.3f, 0.3f, 0.3f);
		p->position = dml::vec3(-3.0f, 0.0f, 3.0f);
		playerModels.push_back(i);
		objects.push_back(std::move(p));
	}

	void createObjTask(tf::Taskflow& tf, const std::string& model, const dml::vec3& scale, const dml::vec4& orientation, const dml::vec3& position) {
		tf.emplace([=] { createObject(model, scale, orientation, position); });
	}

	void loadUniqueObjects() { // load all unqiue objects and all lights
		tfMesh.clear();

		createObjTask(tfMesh, "sword.glb", { 103.2f, 103.2f, 103.2f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f });
		createObjTask(tfMesh, "knight.glb", { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.23f, 0.0f, 2.11f });
		createObjTask(tfMesh, "knight.glb", { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f });
		createObjTask(tfMesh, "sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQ({ 3.0f, 1.0f, -2.11f }, { 0.0f, 0.0f, 0.0f }), { 3.0f, 1.0f, -2.11f });
		createObjTask(tfMesh, "sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQ({ -2.0f, 0.0f, 2.11f }, { 0.0f, 0.0f, 0.0f }), { -2.0f, 0.0f, 2.11f });
		createObjTask(tfMesh, "sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQ({ 0.0f, 2.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }), { 0.0f, 2.0f, 0.0f });

		meshExecutor.run(tfMesh).wait();

		lights.push_back(std::make_unique<Light>(createLight({ -2.0f, 0.0f, -4.0f }, { 0.0f, 1.4f, 0.0f })));
		lights.push_back(std::make_unique<Light>(createLight({ -2.0f, 0.0f, 4.0f }, { 0.0f, 1.4f, 0.0f })));
		lights.push_back(std::make_unique<Light>(createPlayerLight()));

		for (auto& obj : objects) {
			originalObjects.push_back(std::make_unique<dvl::Mesh>(*obj));
		}

		setPlayer(6);
		setPlayer(9);
	}

	bool isRTSupported(VkPhysicalDevice device) {
		// check if extensions are supported
		bool extensionsSupported = checkExtensionSupport(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
			&& checkExtensionSupport(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
			&& checkExtensionSupport(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

		if (!extensionsSupported) {
			return false;
		}

		// check if the device supports ray tracing pipeline features
		VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
		rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
		rtFeatures.pNext = nullptr;

		// check if the device supports acceleration structure features
		VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
		asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		asFeatures.pNext = &rtFeatures;

		VkPhysicalDeviceFeatures2 deviceFeatures2{};
		deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		deviceFeatures2.pNext = &rtFeatures;

		vkGetPhysicalDeviceFeatures2(device, &deviceFeatures2);

		return rtFeatures.rayTracingPipeline == VK_TRUE;
	}

	void createInstance() {
		VkApplicationInfo instanceInfo{};
		instanceInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		instanceInfo.pApplicationName = "My Engine";
		instanceInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		instanceInfo.pEngineName = "No Engine";
		instanceInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		instanceInfo.apiVersion = VK_API_VERSION_1_3;

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
		extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

		const std::vector<const char*> validationLayers = {
			"VK_LAYER_KHRONOS_validation"
		};

		VkInstanceCreateInfo newInfo{};
		newInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		newInfo.pApplicationInfo = &instanceInfo;
		newInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		newInfo.ppEnabledExtensionNames = extensions.data();
		newInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		newInfo.ppEnabledLayerNames = validationLayers.data();

		if (vkCreateInstance(&newInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create instance!");
		}
	}
	int scoreDevice(VkPhysicalDevice device) {
		VkPhysicalDeviceProperties deviceProperties;
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		int score = 0;
		if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			score += 1000;
		}
		else if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
			score += 100;
		}
		return score;
	}

	void pickDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr); // get number of devices
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
		int highestScore = -1;
		for (const auto& device : devices) {
			vkhelper::QueueFamilyIndices indices = vkhelper::findQueueFamilyIndices(surface, device);
			if (indices.allComplete()) {
				VkPhysicalDeviceProperties deviceProperties;
				vkGetPhysicalDeviceProperties(device, &deviceProperties);

				int score = scoreDevice(device);
				if (score > highestScore) {
					bestDevice = device;
					highestScore = score;
				}
			}
		}
		if (bestDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU for graphics, compute, transfer and presentation!");
		}

		// get all of the queue family indices for the best selected device
		queueFamilyIndices = vkhelper::findQueueFamilyIndices(surface, bestDevice);

		// use the best device
		physicalDevice = bestDevice;
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
#ifdef ENABLE_DEBUG
		printCapabilities(deviceProperties);
#else
		utils::sep();
#endif

		// check if ray tracing is supported
		rtSupported = isRTSupported(physicalDevice);
		std::cout << "Raytacing is " << (rtSupported ? "supported" : "not supported") << "!!!!" << std::endl;
		utils::sep();
	}

	void printCapabilities(VkPhysicalDeviceProperties deviceProperties) {
		utils::sep();
		std::cout << "Device Name: " << deviceProperties.deviceName << std::endl;
		std::cout << "Max Descriptor Sets: " << deviceProperties.limits.maxBoundDescriptorSets << std::endl;
		std::cout << "Max Uniform Buffers Descriptors per Set: " << deviceProperties.limits.maxDescriptorSetUniformBuffers << std::endl;
		std::cout << "Max Combined Image Samplers Descriptors per Set: " << deviceProperties.limits.maxDescriptorSetSamplers << std::endl;
		std::cout << "Max Storage Buffers Descriptors per Set: " << deviceProperties.limits.maxDescriptorSetStorageBuffers << std::endl;
		std::cout << "Max Storage Images Descriptors per Set: " << deviceProperties.limits.maxDescriptorSetStorageImages << std::endl;
		std::cout << "Max Input Attachments Descriptors per Set: " << deviceProperties.limits.maxDescriptorSetInputAttachments << std::endl;
		std::cout << "Max Vertex Input Attributes: " << deviceProperties.limits.maxVertexInputAttributes << std::endl;
		std::cout << "Max Vertex Input Bindings: " << deviceProperties.limits.maxVertexInputBindings << std::endl;
		std::cout << "Max Vertex Input Attribute Offset: " << deviceProperties.limits.maxVertexInputAttributeOffset << std::endl;
		std::cout << "Max Vertex Input Binding Stride: " << deviceProperties.limits.maxVertexInputBindingStride << std::endl;
		std::cout << "Max Vertex Output Components: " << deviceProperties.limits.maxVertexOutputComponents << std::endl;
		utils::sep();
	}

	void createLogicalDevice() {
		float queuePriority = 1.0f;
		VkDeviceQueueCreateInfo queueInf{};
		queueInf.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; //creates a structure to hold queue family info
		queueInf.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); // index of the graphics queue family
		queueInf.queueCount = 1;
		queueInf.pQueuePriorities = &queuePriority;

		//VkPhysicalDeviceMaintenance4FeaturesKHR maintenance4{};
		//maintenance4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR;
		//maintenance4.maintenance4 = VK_TRUE;

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.imageCubeArray = VK_TRUE;
		VkPhysicalDeviceDescriptorIndexingFeatures descIndexing{};
		descIndexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
		descIndexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
		descIndexing.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
		descIndexing.runtimeDescriptorArray = VK_TRUE;
		descIndexing.descriptorBindingVariableDescriptorCount = VK_TRUE;
		descIndexing.descriptorBindingPartiallyBound = VK_TRUE;
		//descIndexing.pNext = &maintenance4;

		VkDeviceCreateInfo newInfo{};
		newInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		newInfo.pNext = &descIndexing; // add the indexing features to the pNext chain
		newInfo.pQueueCreateInfos = &queueInf;
		newInfo.queueCreateInfoCount = 1;
		newInfo.pEnabledFeatures = &deviceFeatures; //device features to enable

		// specify the device extensions to enable
		std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
		VK_KHR_MAINTENANCE3_EXTENSION_NAME,
		VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		//VK_EXT_NESTED_COMMAND_BUFFER_EXTENSION_NAME,
		//VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
		};

		if (rtSupported) {
			deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
		}

		for (auto& e : deviceExtensions) {
			if (checkExtensionSupport(e)) {
				std::cout << "---- " << e << " is supported!" << " ----" << std::endl;
			}
			else {
				std::cerr << "---- " << e << " is NOT supported!" << " ----" << std::endl;
				deviceExtensions.erase(std::remove(deviceExtensions.begin(), deviceExtensions.end(), e), deviceExtensions.end());
			}
		}

		newInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		newInfo.ppEnabledExtensionNames = deviceExtensions.data();
		newInfo.enabledLayerCount = 0;
		newInfo.ppEnabledLayerNames = nullptr;
		VkResult result = vkCreateDevice(physicalDevice, &newInfo, nullptr, &device);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("Failed to create logical device!!");
		}

		vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetInstanceProcAddr(instance, "vkCmdPushDescriptorSetKHR");
		if (vkCmdPushDescriptorSetKHR == nullptr) {
			throw std::runtime_error("Failed to get vkCmdPushDescriptorSetKHR function!!!");
		}
		utils::sep();
	}

	bool checkExtensionSupport(const char* extensionName) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

		for (const auto& availableExtension : availableExtensions) {
			if (strcmp(extensionName, availableExtension.extensionName) == 0) {
				return true;
			}
		}
		return false;
	}

	std::vector<char> readFile(const std::string& filename) { //reads shader code from file. it should reads SPIRV binary files
		std::ifstream file(filename, std::ios::ate | std::ios::binary); //ate means start reading at the end of the file and binary means read the file as binary
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file: " + filename);
		}
		size_t fileSize = static_cast<size_t>(file.tellg()); //tellg gets the position of the read/write head
		std::vector<char> buffer(fileSize);
		file.seekg(0); //seekg sets the position of the read/write head
		file.read(buffer.data(), fileSize);
		file.close();
		return buffer;
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void initQueues() {
		vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
		vkGetDeviceQueue(device, queueFamilyIndices.computeFamily.value(), 0, &computeQueue);
		vkGetDeviceQueue(device, queueFamilyIndices.transferFamily.value(), 0, &transferQueue);
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		VkExtent2D actualExtent = { SCREEN_WIDTH, SCREEN_HEIGHT }; //extent=res

		// clamp the width and height to the min and max image extent
		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent; //return the actual extent
	}
	void createSC() {
		vkhelper::SCsupportDetails swapChainSupport = vkhelper::querySCsupport(surface);

		// choose the best surface format, present mode, and swap extent for the swap chain.
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR present = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		// get the number of images for the sc. this is the minumum + 1
		swap.imageCount = swapChainSupport.capabilities.minImageCount + 1;

		// ensure the image count isnt higher than the max image count
		if (swapChainSupport.capabilities.maxImageCount > 0 && swap.imageCount > swapChainSupport.capabilities.maxImageCount) {
			swap.imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		// get the graphics queue family indices
		uint32_t graphicsIndices[] = { queueFamilyIndices.graphicsFamily.value() };

		// create the swap chain.
		VkSwapchainCreateInfoKHR newinfo{};
		newinfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		newinfo.surface = surface;
		newinfo.minImageCount = swap.imageCount;
		newinfo.imageFormat = surfaceFormat.format;
		newinfo.imageColorSpace = surfaceFormat.colorSpace;
		newinfo.imageExtent = extent;
		newinfo.imageArrayLayers = 1; //the num of layers each image has.
		newinfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; //images will be used as color attachment
		newinfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; //image is owned by one queue family at a time and ownership must be explicitly transfered before using it in another queue family
		newinfo.queueFamilyIndexCount = 1;
		newinfo.pQueueFamilyIndices = graphicsIndices;
		newinfo.preTransform = swapChainSupport.capabilities.currentTransform;
		newinfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		newinfo.presentMode = present;
		newinfo.clipped = VK_TRUE; //if the window is obscured, the pixels that are obscured will not be drawn to
		newinfo.oldSwapchain = VK_NULL_HANDLE; //if the swap chain is recreated, the old one is destroyed
		if (vkCreateSwapchainKHR(device, &newinfo, nullptr, &swap.swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		// get the swap chain images
		vkGetSwapchainImagesKHR(device, swap.swapChain, &swap.imageCount, nullptr);
		swap.images.resize(swap.imageCount);
		vkGetSwapchainImagesKHR(device, swap.swapChain, &swap.imageCount, swap.images.data()); //gets the images in the swap chain
		swap.imageFormat = surfaceFormat.format;
		swap.extent = extent;

		// create the image views for the swap chain images
		createSCImageViews();

		// create the viewport for the swap chain
		swapVP.x = 0.0f;
		swapVP.y = 0.0f;
		swapVP.width = static_cast<float>(swap.extent.width); // swap chain extent width for the viewport
		swapVP.height = static_cast<float>(swap.extent.height);
		swapVP.minDepth = 0.0f;
		swapVP.maxDepth = 1.0f;
	}
	VkSurfaceFormatKHR  chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) { //choose the best surface format for the swap chain
		for (const VkSurfaceFormatKHR& format : availableFormats) {
			if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return format;
			}
		}

		return availableFormats[0];

	}
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const VkPresentModeKHR& present : availablePresentModes) {
			if (present == VK_PRESENT_MODE_MAILBOX_KHR) {
				return present;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}
	VkFormat findDepthFormat() {
		std::vector<VkFormat> candidates = {
			VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT //the formats that are supported
		};

		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props); //get the format properties

			if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) { // if the format has the depth stencil attachment bit
				return format;
			}
		}
		throw std::runtime_error("failed to find suitable depth format!");
	}

	void loadSkybox(std::string path) {
		skybox.cubemap.path = SKYBOX_DIR + path;
		createTexturedCubemap(skybox.cubemap);
		vkhelper::createImageView(skybox.cubemap, vkhelper::CUBEMAP);
		vkhelper::createSampler(skybox.cubemap.sampler, skybox.cubemap.mipLevels, vkhelper::CUBEMAP);

		skybox.bufferData.vertexOffset = 0;
		skybox.bufferData.vertexCount = 8;
		skybox.bufferData.indexOffset = 0;
		skybox.bufferData.indexCount = 36;
	}

	void loadMeshTextureData(dvl::Mesh& newObject) {
		modelMtx.lock();

		// load the textures
		if (newObject.material.baseColor.found) {
			createTexturedImage(newObject.material.baseColor, true);
			vkhelper::createImageView(newObject.material.baseColor);
			vkhelper::createSampler(newObject.material.baseColor.sampler, newObject.material.baseColor.mipLevels);

		}

		if (newObject.material.metallicRoughness.found) {
			createTexturedImage(newObject.material.metallicRoughness, true, vkhelper::METALLIC);
			vkhelper::createImageView(newObject.material.metallicRoughness, vkhelper::METALLIC);
			vkhelper::createSampler(newObject.material.metallicRoughness.sampler, newObject.material.metallicRoughness.mipLevels);

		}

		if (newObject.material.normalMap.found) {
			createTexturedImage(newObject.material.normalMap, true, vkhelper::NORMAL);
			vkhelper::createImageView(newObject.material.normalMap, vkhelper::NORMAL);
			vkhelper::createSampler(newObject.material.normalMap.sampler, newObject.material.normalMap.mipLevels);

		}

		if (newObject.material.emissiveMap.found) {
			createTexturedImage(newObject.material.emissiveMap, true, vkhelper::EMISSIVE);
			vkhelper::createImageView(newObject.material.emissiveMap, vkhelper::EMISSIVE);
			vkhelper::createSampler(newObject.material.emissiveMap.sampler, newObject.material.emissiveMap.mipLevels);

		}

		if (newObject.material.occlusionMap.found) {
			createTexturedImage(newObject.material.occlusionMap, true, vkhelper::OCCLUSION);
			vkhelper::createImageView(newObject.material.occlusionMap, vkhelper::OCCLUSION);
			vkhelper::createSampler(newObject.material.occlusionMap.sampler, newObject.material.occlusionMap.mipLevels);

		}

		objects.push_back(std::make_unique<dvl::Mesh>(newObject));
		modelMtx.unlock();

		modelIndex++;
	}


	void loadModel(dml::vec3 scale, dml::vec3 pos, dml::vec4 rot, std::string path) {
		uint32_t meshInd = 0; // index of the mesh in the model

		tinygltf::Model gltfModel;
		tinygltf::TinyGLTF loader;
		std::string err;
		std::string warn;

		bool ret = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, path);
		LOG_WARNING_IF(warn, !warn.empty());

		if (!err.empty()) {
			throw std::runtime_error(err);
		}
		if (!ret) {
			throw std::runtime_error("Failed to load GLTF model");
		}

		// get the index of the parent node for each node
		std::unordered_map<int, int> parentInd;
		for (size_t nodeIndex = 0; nodeIndex < gltfModel.nodes.size(); nodeIndex++) {
			const auto& node = gltfModel.nodes[nodeIndex];
			for (const auto& childIndex : node.children) {
				parentInd[childIndex] = static_cast<int>(nodeIndex);
			}
		}

		// check if the model has any skins or animations (not supported for now)
		LOG_WARNING_IF("The " + path + " contains skinning information", !gltfModel.skins.empty());
		LOG_WARNING_IF("The " + path + " contains animation data", !gltfModel.animations.empty());
		LOG_WARNING_IF("The " + path + " contains cameras", !gltfModel.cameras.empty());

		// check if the gltf model relies on any extensions
		for (const auto& extension : gltfModel.extensionsUsed) {
			LOG_WARNING("The" + path + " relies on: " + extension);
		}

		for (const auto& mesh : gltfModel.meshes) {
			dvl::Mesh m = dvl::loadMesh(mesh, gltfModel, parentInd, meshInd++, scale, pos, rot);
			loadMeshTextureData(m);
		}
	}

	void setupTextures() {
		depthFormat = findDepthFormat();
		static bool init = true;

		// main pass color image
		vkhelper::createImage(mainPassTextures.color.image, mainPassTextures.color.memory, swap.extent.width, swap.extent.height, swap.imageFormat, 1, 1, false, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, commandPool);
		vkhelper::createImageView(mainPassTextures.color, swap.imageFormat);
		vkhelper::createSampler(mainPassTextures.color.sampler, mainPassTextures.color.mipLevels);

		// main pass depth image
		vkhelper::createImage(mainPassTextures.depth.image, mainPassTextures.depth.memory, swap.extent.width, swap.extent.height, depthFormat, 1, 1, false, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		vkhelper::createImageView(mainPassTextures.depth, vkhelper::DEPTH);
		vkhelper::createSampler(mainPassTextures.depth.sampler, mainPassTextures.depth.mipLevels, vkhelper::DEPTH);

		// weighted color image
		vkhelper::createImage(wboit.weightedColor.image, wboit.weightedColor.memory, swap.extent.width, swap.extent.height, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, false, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		vkhelper::createImageView(wboit.weightedColor, VK_FORMAT_R16G16B16A16_SFLOAT);
		vkhelper::createSampler(wboit.weightedColor.sampler, wboit.weightedColor.mipLevels);

		// shadowmaps
		if (init) {
			for (size_t i = 0; i < lights.size(); i++) {
				vkhelper::createImage(lights[i]->shadowMapData.image, lights[i]->shadowMapData.memory, shadowProps.mapWidth, shadowProps.mapHeight, depthFormat, 1, 1, false, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
				vkhelper::createImageView(lights[i]->shadowMapData, vkhelper::DEPTH);
				vkhelper::createSampler(lights[i]->shadowMapData.sampler, lights[i]->shadowMapData.mipLevels, vkhelper::DEPTH);
			}
			init = false;
		}
	}

	void createSCImageViews() { //create the image views for the swap chain images
		swap.imageViews.resize(swap.images.size()); // resize swapChainImageViews to hold all the image views
		for (size_t i = 0; i < swap.images.size(); i++) {
			VkImageViewCreateInfo newinfo{};
			newinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			newinfo.image = swap.images[i]; // assign the current swap chain image
			newinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			newinfo.format = swap.imageFormat;
			newinfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY; // image will maintain its original component ordering
			newinfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			newinfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			newinfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			newinfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // set the aspect mask to color
			newinfo.subresourceRange.baseMipLevel = 0;
			newinfo.subresourceRange.levelCount = 1;
			newinfo.subresourceRange.baseArrayLayer = 0;
			newinfo.subresourceRange.layerCount = 1;
			VkResult result = vkCreateImageView(device, &newinfo, nullptr, &swap.imageViews[i]);
			if (result != VK_SUCCESS) {
				throw std::runtime_error("Failed to create image views!");
			}
		}
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) { //takes in SPIRV binary and creates a shader module
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); //convert the char array to uint32_t array

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}

	void createLightBuffer() {
		lightData.lightCords.resize(lights.size());
		lightData.memSize = lights.size() * sizeof(LightDataObject);

		vkhelper::createBuffer(lightBuffer, lightBufferMem, lightData.lightCords.data(), lightData.memSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	}

	void setupBuffers() {
		vkhelper::createBuffer(instanceBuffer, instanceBufferMem, objInstanceData, sizeof(ModelMatInstanceData), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		vkhelper::createBuffer(cam.buffer, cam.bufferMem, camMatData, sizeof(CamUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		createLightBuffer();

		// skybox buffer data
		vkhelper::createBuffer(skybox.vertBuffer, skybox.vertBufferMem, sizeof(dml::vec3) * skybox.bufferData.vertexCount, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		char* vertexData;
		vkMapMemory(device, skybox.vertBufferMem, 0, sizeof(dml::vec3) * skybox.bufferData.vertexCount, 0, reinterpret_cast<void**>(&vertexData));
		memcpy(vertexData, skybox.vertices.data(), sizeof(dml::vec3) * skybox.bufferData.vertexCount);
		vkUnmapMemory(device, skybox.vertBufferMem);

		vkhelper::createBuffer(skybox.indBuffer, skybox.indBufferMem, sizeof(uint32_t) * skybox.bufferData.indexCount, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		char* indexData;
		vkMapMemory(device, skybox.indBufferMem, 0, sizeof(uint32_t) * skybox.bufferData.indexCount, 0, reinterpret_cast<void**>(&indexData));
		memcpy(indexData, skybox.indices.data(), sizeof(uint32_t) * skybox.bufferData.indexCount);
		vkUnmapMemory(device, skybox.indBufferMem);
	}

	void calcCameraMats() {
		cam.viewMatrix = cam.getViewMatrix();
		cam.projectionMatrix = dml::projection(cam.fov, swap.extent.width / static_cast<float>(swap.extent.height), cam.nearP, cam.farP);
	}

	void calcShadowMats(Light& l) {
		if (l.followPlayer) {
			l.pos = dml::getCamWorldPos(cam.viewMatrix);
			l.target = l.pos + (dml::quatToDir(cam.quat) * -1);
		}
		// spotlight shadow mapping math code
		float aspectRatio = static_cast<float>(shadowProps.mapWidth) / static_cast<float>(shadowProps.mapHeight);
		float nearPlane = 0.01f, farPlane = 100.0f;

		dml::vec3 up = dml::vec3(0.0f, 1.0f, 0.0f);
		if (l.pos == l.target) {
			throw std::runtime_error("Light position and target are the same!");
		}

		l.view = dml::lookAt(l.pos, l.target, up);
		l.proj = dml::spotPerspective(l.outerConeAngle + 15.0f, aspectRatio, nearPlane, farPlane);
	}

	void updateUBO() {
		// calc matrixes for lights
		for (size_t i = 0; i < lights.size(); i++) {
			Light& l = *lights[i];
			calcShadowMats(l);
			copyLightToLightCords(l, lightData.lightCords[i]);
		}

		void* lData;
		vkMapMemory(device, lightBufferMem, 0, lightData.memSize, 0, &lData);
		memcpy(lData, lightData.lightCords.data(), lightData.memSize);
		vkUnmapMemory(device, lightBufferMem);

		// calc matricies for camera
		calcCameraMats();
		memcpy(&camMatData.view, &cam.viewMatrix, sizeof(cam.viewMatrix));
		memcpy(&camMatData.proj, &cam.projectionMatrix, sizeof(cam.projectionMatrix));

		void* cData;
		vkMapMemory(device, cam.bufferMem, 0, sizeof(camMatData), 0, &cData);
		memcpy(cData, &camMatData, sizeof(camMatData));
		vkUnmapMemory(device, cam.bufferMem);

		// calc matrixes for objects
		for (size_t i = 0; i < objects.size(); i++) {
			int render = 0;
			if (objects[i]->player) {
				dml::mat4 t = dml::translate(cam.camPos);
				dml::mat4 r;
				dml::mat4 s = dml::scale(objects[i]->scale);
				dml::mat4 model = (t * r * s) * objects[i]->modelMatrix;
				memcpy(&objInstanceData.object[i].model, &model, sizeof(model));
				render = 1;
				memcpy(&objInstanceData.object[i].render, &render, sizeof(render));
			}
			else {
				memcpy(&objInstanceData.object[i].model, &objects[i]->modelMatrix, sizeof(objects[i]->modelMatrix));
				memcpy(&objInstanceData.object[i].render, &render, sizeof(render));
			}
		}
		void* matrixData;
		vkMapMemory(device, instanceBufferMem, 0, sizeof(objInstanceData), 0, &matrixData);
		memcpy(matrixData, &objInstanceData, sizeof(objInstanceData));
		vkUnmapMemory(device, instanceBufferMem);
	}

	void copyLightToLightCords(const Light& src, LightDataObject& dest) {
		memcpy(&dest.view, &src.view, sizeof(dml::mat4));
		memcpy(&dest.proj, &src.proj, sizeof(dml::mat4));

		memcpy(&dest.pos, &src.pos, sizeof(dml::vec3));
		memcpy(&dest.col, &src.col, sizeof(dml::vec3));
		memcpy(&dest.target, &src.target, sizeof(dml::vec3));
		memcpy(&dest.baseIntensity, &src.baseIntensity, sizeof(float));
		memcpy(&dest.innerConeAngle, &src.innerConeAngle, sizeof(float));
		memcpy(&dest.outerConeAngle, &src.outerConeAngle, sizeof(float));
		memcpy(&dest.constantAttenuation, &src.constantAttenuation, sizeof(float));
		memcpy(&dest.linearAttenuation, &src.linearAttenuation, sizeof(float));
		memcpy(&dest.quadraticAttenuation, &src.quadraticAttenuation, sizeof(float));
	}

	void guiDSLayout() { //descriptor set layout for imgui
		VkDescriptorSetLayoutBinding imguiBinding{};
		imguiBinding.binding = 0;
		imguiBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		imguiBinding.descriptorCount = 1;
		imguiBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // access through the fragment shader

		VkDescriptorSetLayoutCreateInfo imguiLayoutInfo{};
		imguiLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		imguiLayoutInfo.bindingCount = 1;
		imguiLayoutInfo.pBindings = &imguiBinding;

		if (vkCreateDescriptorSetLayout(device, &imguiLayoutInfo, nullptr, &imguiDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create ImGui descriptor set layout!");
		}
	}

	void guiDSPool() { // descriptor pool for imgui
		VkDescriptorPoolSize poolSize{};
		poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSize.descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; // descriptor sets can be freed individually
		poolInfo.maxSets = 1;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create Imgui descriptor pool!");
		}
	}

	std::vector<dvl::Texture> getAllTextures() {
		allTextures.reserve(totalTextureCount);
		size_t currentIndex = 0;
		for (size_t i = 0; i < objects.size(); i++) {
			auto& obj = objects[i];
			if (uniqueModelIndex[obj->meshHash] == i) {
				meshTexStartInd.push_back(static_cast<int>(currentIndex));
				if (obj->material.baseColor.found) {
					allTextures.emplace_back(obj->material.baseColor);
					currentIndex++;
				}
				if (obj->material.metallicRoughness.found) {
					allTextures.emplace_back(obj->material.metallicRoughness);
					currentIndex++;
				}
				if (obj->material.normalMap.found) {
					allTextures.emplace_back(obj->material.normalMap);
					currentIndex++;
				}
				if (obj->material.emissiveMap.found) {
					allTextures.emplace_back(obj->material.emissiveMap);
					currentIndex++;
				}
				if (obj->material.occlusionMap.found) {
					allTextures.emplace_back(obj->material.occlusionMap);
					currentIndex++;
				}
			}
		}
		for (size_t i = 0; i < objects.size(); i++) {
			auto& obj = objects[i];
			if (uniqueModelIndex[obj->meshHash] == i) {
				objects[i]->texIndex = i;
			}
		}
		std::cout << "Finished loading " << totalTextureCount << " textures!" << std::endl;
		return allTextures;
	}

	template<typename Stage>
	void createDSLayoutPool(uint32_t index, VkDescriptorType type, uint32_t size, Stage shaderStage) {
		descs.layouts[index] = vkhelper::createDSLayout(index, type, size, shaderStage);
		descs.pools[index] = vkhelper::createDSPool(type, size);
	}

	void createDS() {
		std::vector<VkDescriptorImageInfo> imageInfos(totalTextureCount);
		for (size_t i = 0; i < totalTextureCount; i++) {
			imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfos[i].imageView = allTextures[i].imageView;
			imageInfos[i].sampler = allTextures[i].sampler;
		}

		uint32_t lightSize = static_cast<uint32_t>(lights.size());
		//uint32_t lightSize = MAX_LIGHTS;

		VkDescriptorBufferInfo lightBufferInfo{};
		lightBufferInfo.buffer = lightBuffer;
		lightBufferInfo.offset = 0;
		lightBufferInfo.range = lightData.memSize;

		shadowInfos.resize(lightSize);
		for (size_t i = 0; i < lightSize; i++) {
			shadowInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			shadowInfos[i].imageView = lights[i]->shadowMapData.imageView;
			shadowInfos[i].sampler = lights[i]->shadowMapData.sampler;
		}

		VkDescriptorImageInfo skyboxInfo{};
		skyboxInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		skyboxInfo.imageView = skybox.cubemap.imageView;
		skyboxInfo.sampler = skybox.cubemap.sampler;

		VkDescriptorBufferInfo camMatBufferInfo{};
		camMatBufferInfo.buffer = cam.buffer;
		camMatBufferInfo.offset = 0;
		camMatBufferInfo.range = sizeof(CamUBO);

		const uint32_t texCompSize = 2;
		std::vector<VkDescriptorImageInfo> compositionPassImageInfo(texCompSize);
		std::array<dvl::Texture, texCompSize> compositionTextures = { mainPassTextures.color, wboit.weightedColor };
		for (size_t i = 0; i < compositionPassImageInfo.size(); i++) {
			compositionPassImageInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			compositionPassImageInfo[i].imageView = compositionTextures[i].imageView;
			compositionPassImageInfo[i].sampler = compositionTextures[i].sampler;
		}

		VkDescriptorImageInfo mainPassDepthInfo{};
		mainPassDepthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		mainPassDepthInfo.imageView = mainPassTextures.depth.imageView;
		mainPassDepthInfo.sampler = mainPassTextures.depth.sampler;

		const uint8_t size = 7;
		descs.sets.resize(size);
		descs.layouts.resize(size);
		descs.pools.resize(size);

		uint32_t texSize = static_cast<uint32_t>(totalTextureCount);

		//initialize descriptor set layouts and pools
		createDSLayoutPool(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texSize, VK_SHADER_STAGE_FRAGMENT_BIT); // array of textures
		createDSLayoutPool(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)); // light data ssbo
		createDSLayoutPool(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_LIGHTS, VK_SHADER_STAGE_FRAGMENT_BIT); // array of shadow map samplers
		createDSLayoutPool(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // 1 sampler for the skybox
		createDSLayoutPool(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT); // camera matricies ubo
		createDSLayoutPool(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texCompSize, VK_SHADER_STAGE_FRAGMENT_BIT); // textures for composition pass
		createDSLayoutPool(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // texture for main pass depth

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorSetCount = 1;

		std::vector<uint32_t> descCountArr = { static_cast<uint32_t>(imageInfos.size()), 1, MAX_LIGHTS, 1, 1, texCompSize, 1 };
		for (uint32_t i = 0; i < descs.sets.size(); i++) {
			VkDescriptorSetVariableDescriptorCountAllocateInfoEXT varCountInfo{};
			varCountInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT;
			varCountInfo.descriptorSetCount = 1;
			varCountInfo.pDescriptorCounts = &descCountArr[i];
			allocInfo.pNext = &varCountInfo; // variableCountInfo is added to the pNext chain

			allocInfo.descriptorPool = descs.pools[i];
			allocInfo.pSetLayouts = &descs.layouts[i];

			VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &descs.sets[i]);
			if (result != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate descriptor sets. Error code: " + std::to_string(result));
			}
		}

		std::array<VkWriteDescriptorSet, size> descriptorWrites{};
		descriptorWrites[0] = vkhelper::createDSWrite(descs.sets[0], 0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageInfos.data(), imageInfos.size());
		descriptorWrites[1] = vkhelper::createDSWrite(descs.sets[1], 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, lightBufferInfo);
		descriptorWrites[2] = vkhelper::createDSWrite(descs.sets[2], 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, shadowInfos.data(), shadowInfos.size());
		descriptorWrites[3] = vkhelper::createDSWrite(descs.sets[3], 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, skyboxInfo);
		descriptorWrites[4] = vkhelper::createDSWrite(descs.sets[4], 4, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, camMatBufferInfo);
		descriptorWrites[5] = vkhelper::createDSWrite(descs.sets[5], 5, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, compositionPassImageInfo.data(), compositionPassImageInfo.size());
		descriptorWrites[6] = vkhelper::createDSWrite(descs.sets[6], 6, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mainPassDepthInfo);

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}

	void setupDescriptorSets(bool initial = true) {
		descs.sets.clear();
		totalTextureCount = 0;
		for (uint32_t i = 0; i < objects.size(); i++) {
			auto& obj = objects[i];
			if (uniqueModelIndex[obj->meshHash] == i) {
				totalTextureCount += obj->textureCount;
			}
		}
		if (initial) {
			getAllTextures();
		}
		createDS(); //create the descriptor set
	}

	void getGLTFImageData(const tinygltf::Image& gltfImage, dvl::Texture& t, unsigned char*& imgData) {
		int width = gltfImage.width; // Set the texture's width, height, and channels
		int height = gltfImage.height;
		int channels = gltfImage.component;


		// delete previously allocated memory if any
		if (imgData != nullptr) {
			delete[] imageData;
		}

		imgData = new unsigned char[width * height * 4]; // create a new array to hold the image data

		// iterate through the image data and copy it into the new array
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				for (int c = 0; c < channels; c++) {
					// copy the data from the original image into the new array
					imgData[(y * width + x) * 4 + c] = gltfImage.image[(y * width + x) * channels + c];
				}
				// if the original image doesn't have an alpha channel, set alpha to 255 (completely opaque)
				if (channels < 4) {
					imgData[(y * width + x) * 4 + 3] = 255;
				}
			}
		}
		imgData = resizeImage(imgData, width, height, t.width, t.height, channels);
	}

	unsigned char* resizeImage(const unsigned char* inputPixels, int originalWidth, int originalHeight,
		int newWidth, int outputHeight, int channels) {
		unsigned char* out = (unsigned char*)malloc(newWidth * outputHeight * channels);
		if (out == NULL) {
			return NULL;  // memory allocation failed
		}

		stbir_resize_uint8(inputPixels, originalWidth, originalHeight, 0,
			out, newWidth, outputHeight, 0,
			channels);

		return out;
	}

	void getImageData(std::string path, unsigned char*& imgData) {
		int texWidth, texHeight, texChannels;
		imgData = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		if (!imgData) {
			throw std::runtime_error("failed to load image!");
		}
	}

	void getImageDataHDR(std::string path, dvl::Texture& t, float*& imgData) {
		int texWidth, texHeight, texChannels;
		imgData = stbi_loadf(path.c_str(), &texWidth, &texHeight, &texChannels, 4); // load RGBA (alpha not used) for the R32G32B32A32_SFLOAT format
		t.width = texWidth;
		t.height = texHeight;
		if (!imgData) {
			std::string error = stbi_failure_reason(); // get the detailed error
			throw std::runtime_error("failed to load HDR image: " + path + "! Reason: " + error);
		}
	}


	void createStagingBuffer(dvl::Texture& tex, bool cubeMap) { // buffer to transfer data from the CPU (imageData) to the GPU sta
		VkBufferCreateInfo bufferInf{};
		auto bpp = cubeMap ? sizeof(float) * 4 : 4;
		VkDeviceSize imageSize = static_cast<VkDeviceSize>(tex.width) * tex.height * bpp;

		if (cubeMap) {
			vkhelper::createBuffer(tex.stagingBuffer, tex.stagingBufferMem, skyboxData, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
		}
		else {
			vkhelper::createBuffer(tex.stagingBuffer, tex.stagingBufferMem, imageData, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
		}
	}

	void createTexturedCubemap(dvl::Texture& tex) {
		getImageDataHDR(tex.path, tex, skyboxData);
		if (skyboxData == nullptr) {
			throw std::runtime_error("failed to load image data!");
		}
		createStagingBuffer(tex, true);

		// clculate the size of one face of the cubemap
		uint32_t faceWidth = tex.width / 4;
		uint32_t faceHeight = tex.height / 3;
		auto bpp = sizeof(float) * 4; // four floats for R32G32B32A32_SFLOAT
		VkDeviceSize faceSize = static_cast<VkDeviceSize>(faceWidth) * faceHeight * bpp;

		// ensure the atlas dimensions are valid for a horizontal cross layout
		if (faceHeight != faceWidth) {
			throw std::runtime_error("Cubemap atlas dimensions are invalid!");
		}

		vkhelper::createImage(tex.image, tex.memory, faceWidth, faceHeight, VK_FORMAT_R32G32B32A32_SFLOAT, 1, 6, true, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

		vkhelper::transitionImageLayout(commandPool, tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, 1, 0);
		VkCommandBuffer copyCmdBuffer = vkhelper::beginSingleTimeCommands(commandPool);

		std::array<VkBufferImageCopy, 6> regions;
		std::array<std::pair<uint32_t, uint32_t>, 6> faceOffsets = {
			 {{2, 1}, // -x
			 {0, 1}, // +x
			 {1, 0}, // +y
			 {1, 2}, // -y
			 {1, 1}, // -z
			 {3, 1}} // +z
		};

		for (uint32_t i = 0; i < regions.size(); i++) {
			VkBufferImageCopy& region = regions[i];

			uint32_t offsetX = faceOffsets[i].first * faceWidth;
			uint32_t offsetY = faceOffsets[i].second * faceHeight;

			region.bufferOffset = offsetY * tex.width * bpp + offsetX * bpp;
			region.bufferRowLength = tex.width;
			region.bufferImageHeight = 0;
			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = i;
			region.imageSubresource.layerCount = 1;

			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = { faceWidth, faceHeight, 1 };

			vkCmdCopyBufferToImage(copyCmdBuffer, tex.stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		}
		vkhelper::endSingleTimeCommands(copyCmdBuffer, commandPool, graphicsQueue);

		vkhelper::transitionImageLayout(commandPool, tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6, 1, 0);
		stbi_image_free(skyboxData);
		skyboxData = nullptr;
	}

	void createTexturedImage(dvl::Texture& tex, bool doMipmap, vkhelper::TextureType type = vkhelper::BASE) {
		if (tex.stagingBuffer == VK_NULL_HANDLE) {
			if (tex.path != "gltf") { // standard image
				getImageData(tex.path, imageData);
			}
			else {
				getGLTFImageData(tex.gltfImage, tex, imageData);
			}
			createStagingBuffer(tex, false);
			tex.mipLevels = doMipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(tex.width, tex.height)))) + 1 : 1;
			VkFormat imgFormat;
			switch (type) {
			case vkhelper::BASE:
				imgFormat = VK_FORMAT_R8G8B8A8_SRGB;
				break;
			case vkhelper::EMISSIVE:
				imgFormat = VK_FORMAT_R8G8B8A8_SRGB;
				break;
			default:
				imgFormat = VK_FORMAT_R8G8B8A8_UNORM;
				break;
			}

			vkhelper::createImage(tex.image, tex.memory, tex.width, tex.height, imgFormat, tex.mipLevels, 1, false, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

			// initialize img and barrier data before buffer copy:
			VkBufferImageCopy region{};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; //specifies the aspect of the image to copy
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount = 1;
			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = { static_cast<uint32_t>(tex.width), static_cast<uint32_t>(tex.height), 1 }; //gets the 2d image extent

			VkCommandBuffer tempBuffer = vkhelper::beginSingleTimeCommands(commandPool);

			vkhelper::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, tex.mipLevels, 0);
			vkCmdCopyBufferToImage(tempBuffer, tex.stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region); //copy the data from the staging buffer to the image

			int mipWidth = tex.width;
			int mipHeight = tex.height;
			if (doMipmap) {
				for (uint32_t j = 0; j < tex.mipLevels; j++) {
					vkhelper::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 1, j);

					if (j < tex.mipLevels - 1) {
						VkImageBlit blit{};
						blit.srcOffsets[0] = { 0, 0, 0 };
						blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
						blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
						blit.srcSubresource.mipLevel = j;
						blit.srcSubresource.baseArrayLayer = 0;
						blit.srcSubresource.layerCount = 1;
						blit.dstOffsets[0] = { 0, 0, 0 };
						blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 }; //if the mip level is greater than 1, divide by 2, otherwise set to 1
						blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
						blit.dstSubresource.mipLevel = j + 1;
						blit.dstSubresource.baseArrayLayer = 0;
						blit.dstSubresource.layerCount = 1;
						vkCmdBlitImage(tempBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
					}

					vkhelper::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, j);

					//for the next mip level, divide the width and height by 2, unless they are already 1
					if (mipWidth > 1) mipWidth /= 2;
					if (mipHeight > 1) mipHeight /= 2;
				}
			}
			else {
				vkhelper::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, tex.mipLevels, 0);
			}

			vkhelper::endSingleTimeCommands(tempBuffer, commandPool, graphicsQueue);
			stbi_image_free(imageData);
			imageData = nullptr;
		}
	}

	static void check_vk_result(VkResult err) { //used to debug imgui errors that have to do with vulkan 
		if (err == 0)
			return;
		std::cerr << "VkResult is " << err << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
		assert(err == 0); //if true, continue, if false, throw error
	}

	void createGraphicsPipeline() {
		std::vector<char> vertShaderCode = readFile(SHADER_DIR + "vertex_shader.spv"); //read the vertex shader binary
		std::vector<char> fragShaderCode = readFile(SHADER_DIR + "fragment_shader.spv");
		vertShaderModule = createShaderModule(vertShaderCode);
		fragShaderModule = createShaderModule(fragShaderCode);
		// shader stage setup 
		VkPipelineShaderStageCreateInfo vertShader{}; //creates a struct for the vertex shader stage info
		vertShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShader.module = vertShaderModule; //assign the vertex shader module
		vertShader.pName = "main";
		VkPipelineShaderStageCreateInfo fragShader{};
		fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShader.module = fragShaderModule;
		fragShader.pName = "main";
		VkPipelineShaderStageCreateInfo stages[] = { vertShader, fragShader }; //create an array of the shader stage structs

		// Vertex input setup (tells Vulkan how to read/organize vertex data based on the stride, offset, and rate)
		VkVertexInputBindingDescription vertBindDesc{};
		vertBindDesc.binding = 0;
		vertBindDesc.stride = sizeof(dvl::Vertex); // Number of bytes from one entry to the next
		vertBindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // The rate when data is loaded

		VkVertexInputBindingDescription instanceBindDesc{};
		instanceBindDesc.binding = 1;
		instanceBindDesc.stride = sizeof(ModelMat);
		instanceBindDesc.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		std::array<VkVertexInputBindingDescription, 2> bindDesc = { vertBindDesc, instanceBindDesc };

		std::vector<VkVertexInputAttributeDescription> attrDesc;
		attrDesc.resize(10);

		attrDesc[0].binding = 0;
		attrDesc[0].location = 0;
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
		attrDesc[0].offset = offsetof(dvl::Vertex, pos);

		// color
		attrDesc[1].binding = 0;
		attrDesc[1].location = 1;
		attrDesc[1].format = VK_FORMAT_R32G32B32A32_SFLOAT; // 4 floats for color
		attrDesc[1].offset = offsetof(dvl::Vertex, col);

		// texture coordinates
		attrDesc[2].binding = 0;
		attrDesc[2].location = 2;
		attrDesc[2].format = VK_FORMAT_R32G32_SFLOAT; // 2 floats for texture coordinates
		attrDesc[2].offset = offsetof(dvl::Vertex, tex);

		// normal
		attrDesc[3].binding = 0;
		attrDesc[3].location = 3;
		attrDesc[3].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for normal
		attrDesc[3].offset = offsetof(dvl::Vertex, normal);

		// tangents
		attrDesc[4].binding = 0;
		attrDesc[4].location = 4;
		attrDesc[4].format = VK_FORMAT_R32G32B32A32_SFLOAT; // 4 floats for tangent
		attrDesc[4].offset = offsetof(dvl::Vertex, tangent);

		// pass the model matrix as a per-instance data
		// seperate the matrix into 4 vec4's so it can be quickly passed and processed
		for (uint32_t i = 0; i < 4; i++) {
			uint8_t index = 5 + i;
			attrDesc[index].binding = 1;
			attrDesc[index].location = index;
			attrDesc[index].format = VK_FORMAT_R32G32B32A32_SFLOAT;
			attrDesc[index].offset = offsetof(ModelMat, model) + sizeof(float) * 4 * i;
		}

		attrDesc[9].binding = 1;
		attrDesc[9].location = 9;
		attrDesc[9].format = VK_FORMAT_R32_UINT; // 1 uint32_t
		attrDesc[9].offset = offsetof(ModelMat, render);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindDesc.size());
		vertexInputInfo.pVertexBindingDescriptions = bindDesc.data();
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size()); // get the size of the attribute description array
		vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data(); // assign the vertex input attribute descriptions

		//input assembly setup (assembles the vertices into primitives)
		VkPipelineInputAssemblyStateCreateInfo inputAssem{}; //create a struct for the input assembly state
		inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; //assign the struct type to the input assembly state
		inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; //set the topology to triangle list (3 vertices per triangle)
		inputAssem.primitiveRestartEnable = VK_FALSE; //if true, then a special index value of 0xFFFF or 0xFFFFFFFF is treated as a restart index

		//viewport and scissors setup (defines the region of the framebuffer that the output will be rendered to)
		VkRect2D scissor{};
		scissor.offset = { 0, 0 }; // top-left corner offset
		scissor.extent = swap.extent;
		VkPipelineViewportStateCreateInfo vpState{};
		vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		vpState.viewportCount = 1;
		vpState.pViewports = &swapVP;
		vpState.scissorCount = 1;
		vpState.pScissors = &scissor;

		//rasterizer setup: Transforms 3D primitives into 3D fragments for display on the screen
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE; //if true, fragments that are beyond the near and far planes are clamped
		rasterizer.rasterizerDiscardEnable = VK_FALSE; //if true, geometry never passes through the rasterizer
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //fill the area of the poly with fragments
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; //cull the back faces of triangle
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_TRUE; //if false, no depth bias is applied to fragments
		rasterizer.depthBiasConstantFactor = 0.0f; //const value that is added to the depth value of a frag
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		//multisampling/anti-aliasing setup: Aggregates multiple samples per pixel, considering alpha values, color, and depth information, and outputs a single colored pixel
		VkPipelineMultisampleStateCreateInfo multiSamp{};
		multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multiSamp.sampleShadingEnable = VK_FALSE;
		multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; //number of samples to use per fragment
		multiSamp.minSampleShading = 1.0f; //min fraction for sample shading; closer to one is smoother
		multiSamp.pSampleMask = nullptr; //array of sample mask values
		multiSamp.alphaToCoverageEnable = VK_TRUE; //enables alpha-to-coverage, blending semi-transparent pixels based on alpha values
		multiSamp.alphaToOneEnable = VK_FALSE; //used for testing right alpha values

		//depth and stencil testing setup: Allows for fragments to be discarded based on depth and stencil values
		VkPipelineDepthStencilStateCreateInfo dStencil{};
		dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		dStencil.depthTestEnable = VK_TRUE; //enable depth testing
		dStencil.depthWriteEnable = VK_TRUE; //enable writing to the depth buffer
		dStencil.depthCompareOp = VK_COMPARE_OP_LESS; //comparison operator that allows for overwriting of new depth values
		dStencil.depthBoundsTestEnable = VK_FALSE; //if true, depth values are clamped to min and max depth bounds
		dStencil.minDepthBounds = 0.0f; //min depth bound
		dStencil.maxDepthBounds = 1.0f;
		dStencil.stencilTestEnable = VK_FALSE; //enable stencil testing
		dStencil.front.failOp = VK_STENCIL_OP_KEEP; //stencil operation to perform if the stencil test fails
		dStencil.front.passOp = VK_STENCIL_OP_KEEP; // stencil operation to perform if the stencil test passes
		dStencil.front.depthFailOp = VK_STENCIL_OP_KEEP; //stencil operation to perform if the stencil test passes, but the depth test fails
		dStencil.front.compareOp = VK_COMPARE_OP_ALWAYS; //comparison operator to use for the stencil test
		dStencil.front.compareMask = 0; // 0 means don't compare against anything
		dStencil.front.writeMask = 0; // 0 means don't write anything to the stencil buffer
		dStencil.front.reference = 0; //reference value to use for the stencil test
		dStencil.back.failOp = VK_STENCIL_OP_KEEP; // what to do if the stencil test fails
		dStencil.back.passOp = VK_STENCIL_OP_KEEP;
		dStencil.back.depthFailOp = VK_STENCIL_OP_KEEP; //what to do if the stencil test passes, but the depth test fails
		dStencil.back.compareOp = VK_COMPARE_OP_ALWAYS;
		dStencil.back.compareMask = 0;
		dStencil.back.writeMask = 0;
		dStencil.back.reference = 0;

		//color blending setup: Combines the output of the fragment shader with the color that is already in the viewbuffer
		VkPipelineColorBlendAttachmentState colorBA{}; //color blend attachment struct
		colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; //color channels to apply the blending operation to
		colorBA.blendEnable = VK_TRUE; //enable blending
		colorBA.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; //blending factors for color channels
		colorBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; //dst is the color already in the framebuffer and src is the color being output from the fragment shader
		colorBA.colorBlendOp = VK_BLEND_OP_ADD; //blending operation to perform
		colorBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; //blending factors for alpha channel
		colorBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBA.alphaBlendOp = VK_BLEND_OP_ADD; //blending operation to perform
		VkPipelineColorBlendStateCreateInfo colorBS{}; //color blend state struct
		colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBS.logicOpEnable = VK_FALSE; //doesnt apply bitwise operation to blending
		colorBS.logicOp = VK_LOGIC_OP_COPY;
		colorBS.attachmentCount = 1; //number of color blend attachments
		colorBS.pAttachments = &colorBA; //array of color blend attachments
		colorBS.blendConstants[0] = 0.0f; //constant values to use in blending operations
		colorBS.blendConstants[1] = 0.0f;
		colorBS.blendConstants[2] = 0.0f;
		colorBS.blendConstants[3] = 0.0f;

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;

		// 1 int for a bitfield of which textures exist, and 1 int for num of textures in a model
		pushConstantRange.size = sizeof(int) * 2;

		//pipeline layout setup: Allows for uniform variables to be passed into the shader
		VkDescriptorSetLayout setLayouts[] = { descs.layouts[0], descs.layouts[1], descs.layouts[2], descs.layouts[4] };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		pipelineLayoutInf.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInf.pushConstantRangeCount = 1;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &mainPassPipeline.layout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!!");
		}

		//render pass setup: Describes the attachments used by the pipeline and how many samples to use for each attachment
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swap.imageFormat; //format of the color attachment
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; //number of samples to use for multisampling
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //what to do with the data in the attachment before rendering
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; //what to do with the data in the attachment after rendering
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; //what to do with the stencil data before rendering
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; //what to do with the stencil data after rendering
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; //layout of the image before the render pass starts
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //layout of the image after the render pass ends

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0; //index of the attachment description in the attachment descriptions array
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; //layout to use for the attachment during the subpass

		// define depth attachment:
		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = depthFormat;
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; //type of pipeline to bind to
		subpass.colorAttachmentCount = 1; //number of color attachments
		subpass.pColorAttachments = &colorAttachmentRef; //array of color attachment references
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		//define the render pass
		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
		VkRenderPassCreateInfo renderPassInf{};
		renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInf.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInf.pAttachments = attachments.data();
		renderPassInf.subpassCount = 1; //number of subpasses
		renderPassInf.pSubpasses = &subpass; //array of subpasses
		VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, &mainPassPipeline.renderPass);
		if (renderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

		//pipeline setup: Describes the pipeline to be created
		VkGraphicsPipelineCreateInfo pipelineInf{};
		pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInf.stageCount = 2; // Vertex and fragment shaders
		pipelineInf.pStages = stages;
		pipelineInf.pVertexInputState = &vertexInputInfo;
		pipelineInf.pInputAssemblyState = &inputAssem;
		pipelineInf.pViewportState = &vpState;
		pipelineInf.pRasterizationState = &rasterizer;
		pipelineInf.pMultisampleState = &multiSamp;
		pipelineInf.pDepthStencilState = &dStencil;
		pipelineInf.pColorBlendState = &colorBS;
		pipelineInf.layout = mainPassPipeline.layout;
		pipelineInf.renderPass = mainPassPipeline.renderPass;
		pipelineInf.subpass = 0;
		pipelineInf.basePipelineHandle = VK_NULL_HANDLE; // no base pipeline for now
		pipelineInf.basePipelineIndex = -1;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &mainPassPipeline.graphicsPipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}
	}

	void createShadowPipeline() {
		// get shader data
		auto vertShaderSPV = readFile(SHADER_DIR + "shadow_vert_shader.spv");
		auto fragShaderSPV = readFile(SHADER_DIR + "shadow_frag_shader.spv");
		VkShaderModule shadowVertShaderModule = createShaderModule(vertShaderSPV);
		VkShaderModule shadowFragShaderModule = createShaderModule(fragShaderSPV);

		VkPipelineShaderStageCreateInfo vertStage{};
		vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertStage.module = shadowVertShaderModule;
		vertStage.pName = "main";
		VkPipelineShaderStageCreateInfo fragStage{};
		fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragStage.module = shadowFragShaderModule;
		fragStage.pName = "main";
		VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

		VkVertexInputBindingDescription vertBindDesc{};
		vertBindDesc.binding = 0;
		vertBindDesc.stride = sizeof(dvl::Vertex);
		vertBindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputBindingDescription instanceBindDesc{};
		instanceBindDesc.binding = 1;
		instanceBindDesc.stride = sizeof(ModelMat);
		instanceBindDesc.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		std::array<VkVertexInputBindingDescription, 2> bindDesc = { vertBindDesc, instanceBindDesc };

		std::vector<VkVertexInputAttributeDescription> attrDesc; // array of attribute descriptions
		attrDesc.resize(5);

		// vertex position attribute
		attrDesc[0].binding = 0;
		attrDesc[0].location = 0;
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
		attrDesc[0].offset = offsetof(dvl::Vertex, pos);

		// pass the model matrix as a per-instance data
		// seperate the matrix into 4 vec4's so it can be quickly passed and processed
		for (uint32_t i = 0; i < 4; i++) {
			uint8_t index = i + 1;
			attrDesc[index].binding = 1;
			attrDesc[index].location = index;
			attrDesc[index].format = VK_FORMAT_R32G32B32A32_SFLOAT;
			attrDesc[index].offset = offsetof(ModelMat, model) + sizeof(float) * 4 * i;
		}

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindDesc.size());
		vertexInputInfo.pVertexBindingDescriptions = bindDesc.data();
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size()); // get the size of the attribute description array
		vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data(); // assign the vertex input attribute descriptions

		VkPipelineInputAssemblyStateCreateInfo inputAssem{};
		inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssem.primitiveRestartEnable = VK_FALSE;

		VkViewport shadowVP{}; // shadow viewport
		shadowVP.x = 0.0f;
		shadowVP.y = 0.0f;
		shadowVP.width = static_cast<float>(shadowProps.mapWidth);
		shadowVP.height = static_cast<float>(shadowProps.mapHeight);
		shadowVP.minDepth = 0.0f;
		shadowVP.maxDepth = 1.0f;

		VkRect2D shadowScissor{}; // shadow scissor
		shadowScissor.offset = { 0, 0 };
		shadowScissor.extent.width = shadowProps.mapWidth;
		shadowScissor.extent.height = shadowProps.mapHeight;

		VkPipelineViewportStateCreateInfo vpState{}; // viewport state
		vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		vpState.viewportCount = 1;
		vpState.pViewports = &shadowVP;
		vpState.scissorCount = 1;
		vpState.pScissors = &shadowScissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_TRUE;
		rasterizer.depthBiasConstantFactor = 1.75f;
		rasterizer.depthBiasSlopeFactor = 1.75f;
		rasterizer.depthBiasClamp = 0.0f;


		VkPipelineMultisampleStateCreateInfo multiSamp{};
		multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multiSamp.sampleShadingEnable = VK_FALSE;
		multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// creating the depth stencil state
		VkPipelineDepthStencilStateCreateInfo dStencil{};
		dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		dStencil.depthTestEnable = VK_TRUE; //enable depth test
		dStencil.depthWriteEnable = VK_TRUE;
		dStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		dStencil.depthBoundsTestEnable = VK_FALSE;
		dStencil.minDepthBounds = 0.0f;
		dStencil.maxDepthBounds = 1.0f;
		dStencil.stencilTestEnable = VK_FALSE;

		// creating the shadow map render pass
		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = depthFormat;
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		// create the depth attachment reference
		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 0;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// creatr the subpass
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 0;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		// create the render pass
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &depthAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadowMapPipeline.renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow map render pass!");
		}

		VkPipelineColorBlendStateCreateInfo colorBS{};
		colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBS.attachmentCount = 0;

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int); // 1 int for the light index

		VkDescriptorSetLayout setLayouts[] = { descs.layouts[1] }; // the light data ssbo
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = 1;
		pipelineLayoutInf.pSetLayouts = setLayouts;
		pipelineLayoutInf.pushConstantRangeCount = 1; // one range of push constants
		pipelineLayoutInf.pPushConstantRanges = &pushConstantRange;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &shadowMapPipeline.layout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!!");
		}

		// create the pipeline based off this pipeline and some data from the main pipeline
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = stages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssem;
		pipelineInfo.pViewportState = &vpState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multiSamp;
		pipelineInfo.pDepthStencilState = &dStencil;
		pipelineInfo.pColorBlendState = &colorBS;
		pipelineInfo.layout = shadowMapPipeline.layout;
		pipelineInfo.renderPass = shadowMapPipeline.renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowMapPipeline.graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow map pipeline!!!!");
		}
	}

	void createSkyboxPipeline() { // same as the normal pipeline, but with a few small changes
		std::vector<char> vertShaderCode = readFile(SHADER_DIR + "sky_vert_shader.spv");
		std::vector<char> fragShaderCode = readFile(SHADER_DIR + "sky_frag_shader.spv");
		vertShaderModule = createShaderModule(vertShaderCode);
		fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShader{};
		vertShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShader.module = vertShaderModule;
		vertShader.pName = "main";
		VkPipelineShaderStageCreateInfo fragShader{};
		fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShader.module = fragShaderModule;
		fragShader.pName = "main";
		VkPipelineShaderStageCreateInfo stages[] = { vertShader, fragShader };

		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding = 0;
		bindDesc.stride = sizeof(dml::vec3); // the stride is the size of vec3
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 1> attrDesc;

		attrDesc[0].binding = 0;
		attrDesc[0].location = 0;
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // vec3 is three 32-bit floats
		attrDesc[0].offset = 0; // offset within the vec3 is 0, since pos is the first element

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindDesc;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
		vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssem{};
		inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssem.primitiveRestartEnable = VK_FALSE;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swap.extent;
		VkPipelineViewportStateCreateInfo vpState{};
		vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		vpState.viewportCount = 1;
		vpState.pViewports = &swapVP;
		vpState.scissorCount = 1;
		vpState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multiSamp{};
		multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multiSamp.sampleShadingEnable = VK_FALSE;
		multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multiSamp.minSampleShading = 1.0f;
		multiSamp.pSampleMask = nullptr;
		multiSamp.alphaToCoverageEnable = VK_FALSE;
		multiSamp.alphaToOneEnable = VK_FALSE;

		VkPipelineDepthStencilStateCreateInfo dStencil{};
		dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		dStencil.depthTestEnable = VK_TRUE; //enable depth test
		dStencil.depthWriteEnable = VK_FALSE; // dont write to the depth buffer
		dStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		dStencil.depthBoundsTestEnable = VK_FALSE;
		dStencil.minDepthBounds = 0.0f;
		dStencil.maxDepthBounds = 1.0f;
		dStencil.stencilTestEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBA{};
		colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBA.blendEnable = VK_FALSE; // disable blending for the skybox
		VkPipelineColorBlendStateCreateInfo colorBS{};
		colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBS.logicOpEnable = VK_FALSE;
		colorBS.logicOp = VK_LOGIC_OP_COPY;
		colorBS.attachmentCount = 1;
		colorBS.pAttachments = &colorBA;

		VkDescriptorSetLayout setLayouts[] = { descs.layouts[3], descs.layouts[4] };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &skybox.pipelineLayout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout for skybox!");
		}

		VkGraphicsPipelineCreateInfo pipelineInf{};
		pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInf.stageCount = 2;
		pipelineInf.pStages = stages;
		pipelineInf.pVertexInputState = &vertexInputInfo;
		pipelineInf.pInputAssemblyState = &inputAssem;
		pipelineInf.pViewportState = &vpState;
		pipelineInf.pRasterizationState = &rasterizer;
		pipelineInf.pMultisampleState = &multiSamp;
		pipelineInf.pDepthStencilState = &dStencil;
		pipelineInf.pColorBlendState = &colorBS;
		pipelineInf.layout = skybox.pipelineLayout;
		pipelineInf.renderPass = mainPassPipeline.renderPass;
		pipelineInf.subpass = 0;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &skybox.pipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline for skybox!");
		}
	}

	void createWBOITPipeline() {
		std::vector<char> vertShaderCode = readFile(SHADER_DIR + "wboit_vert_shader.spv");
		std::vector<char> fragShaderCode = readFile(SHADER_DIR + "wboit_frag_shader.spv");
		vertShaderModule = createShaderModule(vertShaderCode);
		fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShader{};
		vertShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShader.module = vertShaderModule;
		vertShader.pName = "main";
		VkPipelineShaderStageCreateInfo fragShader{};
		fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShader.module = fragShaderModule;
		fragShader.pName = "main";
		VkPipelineShaderStageCreateInfo stages[] = { vertShader, fragShader };

		VkVertexInputBindingDescription vertBindDesc{};
		vertBindDesc.binding = 0;
		vertBindDesc.stride = sizeof(dvl::Vertex);
		vertBindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputBindingDescription instanceBindDesc{};
		instanceBindDesc.binding = 1;
		instanceBindDesc.stride = sizeof(ModelMat);
		instanceBindDesc.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		std::array<VkVertexInputBindingDescription, 2> bindDesc = { vertBindDesc, instanceBindDesc };

		std::vector<VkVertexInputAttributeDescription> attrDesc;

		attrDesc.resize(10);

		attrDesc[0].binding = 0;
		attrDesc[0].location = 0;
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
		attrDesc[0].offset = offsetof(dvl::Vertex, pos);

		// color
		attrDesc[1].binding = 0;
		attrDesc[1].location = 1;
		attrDesc[1].format = VK_FORMAT_R32G32B32A32_SFLOAT; // 4 floats for color
		attrDesc[1].offset = offsetof(dvl::Vertex, col);

		// texture coordinates
		attrDesc[2].binding = 0;
		attrDesc[2].location = 2;
		attrDesc[2].format = VK_FORMAT_R32G32_SFLOAT; // 2 floats for texture coordinates
		attrDesc[2].offset = offsetof(dvl::Vertex, tex);

		// normal
		attrDesc[3].binding = 0;
		attrDesc[3].location = 3;
		attrDesc[3].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for normal
		attrDesc[3].offset = offsetof(dvl::Vertex, normal);

		// tangents
		attrDesc[4].binding = 0;
		attrDesc[4].location = 4;
		attrDesc[4].format = VK_FORMAT_R32G32B32A32_SFLOAT; // 4 floats for tangent
		attrDesc[4].offset = offsetof(dvl::Vertex, tangent);

		// pass the model matrix as a per-instance data
		for (uint32_t i = 0; i < 4; i++) {
			uint8_t index = 5 + i;
			attrDesc[index].binding = 1;
			attrDesc[index].location = index;
			attrDesc[index].format = VK_FORMAT_R32G32B32A32_SFLOAT;
			attrDesc[index].offset = offsetof(ModelMat, model) + sizeof(float) * 4 * i;
		}

		attrDesc[9].binding = 1;
		attrDesc[9].location = 9;
		attrDesc[9].format = VK_FORMAT_R32_UINT; // 1 uint32_t
		attrDesc[9].offset = offsetof(ModelMat, render);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindDesc.size());
		vertexInputInfo.pVertexBindingDescriptions = bindDesc.data();
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
		vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssem{};
		inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssem.primitiveRestartEnable = VK_FALSE;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swap.extent;
		VkPipelineViewportStateCreateInfo vpState{};
		vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		vpState.viewportCount = 1;
		vpState.pViewports = &swapVP;
		vpState.scissorCount = 1;
		vpState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multiSamp{};
		multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multiSamp.sampleShadingEnable = VK_FALSE;
		multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multiSamp.minSampleShading = 1.0f;
		multiSamp.pSampleMask = nullptr;
		multiSamp.alphaToCoverageEnable = VK_FALSE;
		multiSamp.alphaToOneEnable = VK_FALSE;

		VkPipelineDepthStencilStateCreateInfo dStencil{};
		dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		dStencil.depthTestEnable = VK_TRUE; //enable depth test
		dStencil.depthWriteEnable = VK_FALSE; //dont write to the depth buffer
		dStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		dStencil.depthBoundsTestEnable = VK_FALSE;
		dStencil.minDepthBounds = 0.0f;
		dStencil.maxDepthBounds = 1.0f;
		dStencil.stencilTestEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBA{};
		colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBA.blendEnable = VK_TRUE;
		colorBA.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBA.dstColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBA.colorBlendOp = VK_BLEND_OP_ADD;
		colorBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBA.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBS{};
		colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBS.logicOpEnable = VK_FALSE;
		colorBS.logicOp = VK_LOGIC_OP_COPY;
		colorBS.attachmentCount = 1;
		colorBS.pAttachments = &colorBA;

		// weighted color
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		// define the render pass
		VkRenderPassCreateInfo renderPassInf{};
		renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInf.attachmentCount = 1;
		renderPassInf.pAttachments = &colorAttachment;
		renderPassInf.subpassCount = 1;
		renderPassInf.pSubpasses = &subpass;
		VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, &wboitPipeline.renderPass);
		if (renderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int) * 2;

		VkDescriptorSetLayout setLayouts[] = { descs.layouts[0], descs.layouts[1], descs.layouts[2], descs.layouts[4], descs.layouts[6] };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		pipelineLayoutInf.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInf.pushConstantRangeCount = 1;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &wboitPipeline.layout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout for WBOIT!!");
		}

		VkGraphicsPipelineCreateInfo pipelineInf{};
		pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInf.stageCount = 2;
		pipelineInf.pStages = stages;
		pipelineInf.pVertexInputState = &vertexInputInfo;
		pipelineInf.pInputAssemblyState = &inputAssem;
		pipelineInf.pViewportState = &vpState;
		pipelineInf.pRasterizationState = &rasterizer;
		pipelineInf.pMultisampleState = &multiSamp;
		pipelineInf.pDepthStencilState = &dStencil;
		pipelineInf.pColorBlendState = &colorBS;
		pipelineInf.layout = wboitPipeline.layout;
		pipelineInf.renderPass = wboitPipeline.renderPass;
		pipelineInf.subpass = 0;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &wboitPipeline.graphicsPipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline for WBOIT!");
		}
	}

	void createCompositionPipeline() {
		std::vector<char> vertShaderCode = readFile(SHADER_DIR + "composition_vert_shader.spv");
		std::vector<char> fragShaderCode = readFile(SHADER_DIR + "composition_frag_shader.spv");
		vertShaderModule = createShaderModule(vertShaderCode);
		fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShader{};
		vertShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShader.module = vertShaderModule;
		vertShader.pName = "main";
		VkPipelineShaderStageCreateInfo fragShader{};
		fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShader.module = fragShaderModule;
		fragShader.pName = "main";
		VkPipelineShaderStageCreateInfo stages[] = { vertShader, fragShader };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr;
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr;

		VkPipelineInputAssemblyStateCreateInfo inputAssem{};
		inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssem.primitiveRestartEnable = VK_FALSE;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swap.extent;
		VkPipelineViewportStateCreateInfo vpState{};
		vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		vpState.viewportCount = 1;
		vpState.pViewports = &swapVP;
		vpState.scissorCount = 1;
		vpState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE; // no culling
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multiSamp{};
		multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multiSamp.sampleShadingEnable = VK_FALSE;
		multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multiSamp.minSampleShading = 1.0f;
		multiSamp.pSampleMask = nullptr;
		multiSamp.alphaToCoverageEnable = VK_FALSE;
		multiSamp.alphaToOneEnable = VK_FALSE;

		VkPipelineDepthStencilStateCreateInfo dStencil{};
		dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		dStencil.depthTestEnable = VK_FALSE; // no depth write or test
		dStencil.depthWriteEnable = VK_FALSE;
		dStencil.depthBoundsTestEnable = VK_FALSE;
		dStencil.minDepthBounds = 0.0f;
		dStencil.maxDepthBounds = 1.0f;
		dStencil.stencilTestEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBA{};
		colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBA.blendEnable = VK_TRUE;
		colorBA.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBA.colorBlendOp = VK_BLEND_OP_ADD;
		colorBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBA.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBS{};
		colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBS.logicOpEnable = VK_FALSE;
		colorBS.logicOp = VK_LOGIC_OP_COPY;
		colorBS.attachmentCount = 1;
		colorBS.pAttachments = &colorBA;

		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swap.imageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkRenderPassCreateInfo renderPassInf{};
		renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInf.attachmentCount = 1;
		renderPassInf.pAttachments = &colorAttachment;
		renderPassInf.subpassCount = 1;
		renderPassInf.pSubpasses = &subpass;
		VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, &compPipelineData.renderPass);
		if (renderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

		VkDescriptorSetLayout setLayouts[] = { descs.layouts[5] };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &compPipelineData.layout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout for composition!!");
		}

		VkGraphicsPipelineCreateInfo pipelineInf{};
		pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInf.stageCount = 2;
		pipelineInf.pStages = stages;
		pipelineInf.pVertexInputState = &vertexInputInfo;
		pipelineInf.pInputAssemblyState = &inputAssem;
		pipelineInf.pViewportState = &vpState;
		pipelineInf.pRasterizationState = &rasterizer;
		pipelineInf.pMultisampleState = &multiSamp;
		pipelineInf.pDepthStencilState = &dStencil;
		pipelineInf.pColorBlendState = &colorBS;
		pipelineInf.layout = compPipelineData.layout;
		pipelineInf.renderPass = compPipelineData.renderPass;
		pipelineInf.subpass = 0;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &compPipelineData.graphicsPipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline for the composition pass!!!!!!!!");
		}
	}


	void setupPipelines(bool shadow) {
		createGraphicsPipeline();
		createCompositionPipeline();
		createSkyboxPipeline();
		if (shadow) createShadowPipeline();
		createWBOITPipeline();
	}

	void imguiSetup() {
		// descriptor set creation for the gui:
		guiDSLayout();
		guiDSPool();

		// imgui setup:
		uint32_t graphicsQueueFamily = queueFamilyIndices.graphicsFamily.value();
		ImGui_ImplVulkan_InitInfo initInfo = {};
		initInfo.Instance = instance;
		initInfo.PhysicalDevice = physicalDevice;
		initInfo.Device = device;
		initInfo.QueueFamily = graphicsQueueFamily;
		initInfo.Queue = graphicsQueue;
		initInfo.PipelineCache = VK_NULL_HANDLE; // no pipeline cache for now
		initInfo.DescriptorPool = imguiDescriptorPool;
		initInfo.Allocator = VK_NULL_HANDLE;
		initInfo.MinImageCount = swap.imageCount;
		initInfo.ImageCount = swap.imageCount;
		initInfo.CheckVkResultFn = check_vk_result; // function to check vulkan results
		ImGui_ImplVulkan_Init(&initInfo, compPipelineData.renderPass);

		// upload fonts, etc:
		VkCommandPool guiCommandPool = createCommandPool();
		VkCommandBuffer guiCommandBuffer = vkhelper::beginSingleTimeCommands(guiCommandPool);
		ImGui_ImplVulkan_CreateFontsTexture(guiCommandBuffer);
		vkhelper::endSingleTimeCommands(guiCommandBuffer, guiCommandPool, graphicsQueue);
		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}

	void setupFences() {
		inFlightFences.resize(swap.images.size());
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // signaled state fence (fence is signaled when created)
		for (size_t i = 0; i < inFlightFences.size(); i++) {
			if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	VkCommandPool createCommandPool() {
		VkCommandPool cPool;
		VkCommandPoolCreateInfo poolInf{};
		poolInf.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInf.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); //the queue family that will be using this command pool
		poolInf.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // enable reset command buffer flag
		VkResult result = vkCreateCommandPool(device, &poolInf, nullptr, &cPool);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
		return cPool;
	}

	void allocateCommandBuffers(CommandBufferSet& cmdBuffers, size_t primaryCount, size_t secondaryCount = 0) {
		cmdBuffers.primary.resize(primaryCount);

		for (size_t i = 0; i < primaryCount; i++) {
			cmdBuffers.primary.pools[i] = createCommandPool();
			cmdBuffers.primary.buffers[i] = vkhelper::allocateCommandBuffers(cmdBuffers.primary.pools[i]);
		}

		if (secondaryCount) {
			cmdBuffers.secondary.resize(secondaryCount);
			cmdBuffers.secondary.resize(secondaryCount);

			for (size_t i = 0; i < secondaryCount; i++) {
				cmdBuffers.secondary.pools[i] = createCommandPool();
				cmdBuffers.secondary.buffers[i] = vkhelper::allocateCommandBuffers(cmdBuffers.secondary.pools[i], VK_COMMAND_BUFFER_LEVEL_SECONDARY);
			}
		}
	}

	void createCommandBuffers() {
		allocateCommandBuffers(shadowMapCommandBuffers, lights.size(), lights.size());
		allocateCommandBuffers(mainPassCommandBuffers, swap.imageCount, 1);
		allocateCommandBuffers(wboitCommandBuffers, swap.imageCount, 1);
		allocateCommandBuffers(compCommandBuffers, swap.imageCount, 1);
	}

	void createModelBuffers() { // creates the vertex and index buffers for the unique models into a single buffer
		std::sort(objects.begin(), objects.end(), [](const auto& a, const auto& b) { return a->meshHash < b->meshHash; });

		bufferData.resize(getUniqueModels());
		uniqueModelIndex.clear();
		modelHashToBufferIndex.clear();

		VkDeviceSize totalVertexBufferSize = 0;
		VkDeviceSize totalIndexBufferSize = 0;

		// get the total size of the vertex and index buffers
		uint32_t ind = 0;
		for (size_t i = 0; i < objects.size(); i++) {
			auto& obj = objects[i];
			if (uniqueModelIndex.find(obj->meshHash) == uniqueModelIndex.end()) {
				totalVertexBufferSize += sizeof(dvl::Vertex) * obj->vertices.size();
				totalIndexBufferSize += sizeof(uint32_t) * obj->indices.size();
				uniqueModelIndex[obj->meshHash] = i; //store the index of the object
				modelHashToBufferIndex[obj->meshHash] = ind++;
			}
		}

		// create and map the vertex buffer
		vkhelper::createBuffer(vertBuffer, vertBufferMem, totalVertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT); // create the combined vertex buffer
		char* vertexData;
		vkMapMemory(device, vertBufferMem, 0, totalVertexBufferSize, 0, reinterpret_cast<void**>(&vertexData));
		VkDeviceSize currentVertexOffset = 0;

		// create and map the index buffer
		vkhelper::createBuffer(indBuffer, indBufferMem, totalIndexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT); // create the combined index buffer
		char* indexData;
		vkMapMemory(device, indBufferMem, 0, totalIndexBufferSize, 0, reinterpret_cast<void**>(&indexData));
		VkDeviceSize currentIndexOffset = 0;

		for (size_t i = 0; i < objects.size(); i++) {
			size_t modelInd = uniqueModelIndex[objects[i]->meshHash];
			if (modelInd != i) continue; // skip if not the first instance of the model
			size_t bufferInd = modelHashToBufferIndex[objects[i]->meshHash];

			// vertex data
			bufferData[bufferInd].vertexOffset = static_cast<uint32_t>(currentVertexOffset);
			bufferData[bufferInd].vertexCount = static_cast<uint32_t>(objects[modelInd]->vertices.size());
			memcpy(vertexData, objects[modelInd]->vertices.data(), bufferData[bufferInd].vertexCount * sizeof(dvl::Vertex));
			vertexData += bufferData[bufferInd].vertexCount * sizeof(dvl::Vertex);
			currentVertexOffset += bufferData[bufferInd].vertexCount;

			// index data
			bufferData[bufferInd].indexOffset = static_cast<uint32_t>(currentIndexOffset);
			bufferData[bufferInd].indexCount = static_cast<uint32_t>(objects[modelInd]->indices.size());
			memcpy(indexData, objects[modelInd]->indices.data(), bufferData[bufferInd].indexCount * sizeof(uint32_t));
			indexData += bufferData[bufferInd].indexCount * sizeof(uint32_t);
			currentIndexOffset += bufferData[bufferInd].indexCount;
		}
		vkUnmapMemory(device, vertBufferMem);
		vkUnmapMemory(device, indBufferMem);
	}

	void cloneObject(dml::vec3 pos, uint16_t object, dml::vec3 scale, dml::vec4 rotation) {
		auto m = std::make_unique<dvl::Mesh>(*originalObjects[object]);

		m->scale = scale;
		m->position = pos;
		m->startObj = false;
		m->rotation = rotation;

		dml::mat4 newModel = dml::translate(pos) * dml::rotateQ(rotation) * dml::scale(scale);
		m->modelMatrix = newModel * m->modelMatrix;
		objects.push_back(std::move(m));
	}

	uint32_t getModelNumHash(size_t hash) { // get the number of models that have the same hash
		uint32_t count = 0;
		for (auto& m : objects) {
			if (m->meshHash == hash) {
				count++;
			}
		}
		return count;
	}

	size_t getUniqueModels() { // get the number of unique models
		std::unordered_set<size_t> uniqueModels;
		for (auto& m : objects) {
			uniqueModels.insert(m->meshHash);
		}
		return uniqueModels.size();
	}

	void recreateModelBuffers() {
		vkDestroyBuffer(device, vertBuffer, nullptr);
		vkFreeMemory(device, vertBufferMem, nullptr);
		vkDestroyBuffer(device, indBuffer, nullptr);
		vkFreeMemory(device, indBufferMem, nullptr);
		createModelBuffers();
	}

	void summonModel() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		dml::vec3 pos = dml::getCamWorldPos(cam.viewMatrix);

		cloneObject(pos, 6, { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });
		cloneObject(pos, 9, { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });

		recreateModelBuffers();
	}

	void updateLightDS() {
		uint32_t lightSize = static_cast<uint32_t>(lights.size());

		recreateLightBuffer();
		VkDescriptorBufferInfo lightBufferInfo{};
		lightBufferInfo.buffer = lightBuffer;
		lightBufferInfo.offset = 0;
		lightBufferInfo.range = lightData.memSize;

		std::array<VkWriteDescriptorSet, 2> dw{};
		dw[0] = vkhelper::createDSWrite(descs.sets[1], 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, lightBufferInfo);
		dw[1] = vkhelper::createDSWrite(descs.sets[2], 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, shadowInfos.data(), lightSize);

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(dw.size()), dw.data(), 0, nullptr);
	}

	void summonLight() {
		if (lights.size() + 1 > MAX_LIGHTS) return;
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		dml::vec3 pos = dml::getCamWorldPos(cam.viewMatrix);
		dml::vec3 target = pos + (dml::quatToDir(cam.quat) * -1);
		Light l = createLight(pos, target);

		vkhelper::createImage(l.shadowMapData.image, l.shadowMapData.memory, shadowProps.mapWidth, shadowProps.mapHeight, depthFormat, 1, 1, false, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		vkhelper::createImageView(l.shadowMapData, vkhelper::DEPTH);
		vkhelper::createSampler(l.shadowMapData.sampler, l.shadowMapData.mipLevels, vkhelper::DEPTH);

		vkhelper::createFB(shadowMapPipeline.renderPass, l.frameBuffer, &l.shadowMapData.imageView, 1, shadowProps.mapWidth, shadowProps.mapHeight);

		VkCommandPool p = createCommandPool();
		VkCommandBuffer c = vkhelper::allocateCommandBuffers(p);
		shadowMapCommandBuffers.primary.buffers.push_back(c);
		shadowMapCommandBuffers.primary.pools.push_back(p);

		VkCommandPool p2 = createCommandPool();
		VkCommandBuffer c2 = vkhelper::allocateCommandBuffers(p2, VK_COMMAND_BUFFER_LEVEL_SECONDARY);

		shadowMapCommandBuffers.secondary.buffers.push_back(c2);
		shadowMapCommandBuffers.secondary.pools.push_back(p2);

		lights.push_back(std::make_unique<Light>(l));

		VkDescriptorImageInfo shadowInfo{};
		shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		shadowInfo.imageView = lights.back()->shadowMapData.imageView;
		shadowInfo.sampler = lights.back()->shadowMapData.sampler;
		shadowInfos.push_back(shadowInfo);

		updateLightDS();
	}

	void recreateLightBuffer() {
		vkDestroyBuffer(device, lightBuffer, nullptr);
		vkFreeMemory(device, lightBufferMem, nullptr);
		createLightBuffer();
	}

	void recordMainSecondaryCmdBuffers(VkCommandBuffer& secondary, const PipelineData& pipe, const VkCommandBufferBeginInfo& beginInfo, const VkDescriptorSet* descriptorsets, const size_t descriptorCount, const bool startCommand, const bool endCommand) {
		const std::array<VkBuffer, 2> vertexBuffersArray = { vertBuffer, instanceBuffer };
		const std::array<VkDeviceSize, 2> offsets = { 0, 0 };

		if (startCommand) {
			if (vkBeginCommandBuffer(secondary, &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording main secondary command buffer!");
			}
		}

		vkCmdBindPipeline(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.graphicsPipeline);
		vkCmdBindDescriptorSets(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, static_cast<uint32_t>(descriptorCount), descriptorsets, 0, nullptr);

		// bind the vertex and instance buffers
		vkCmdBindVertexBuffers(secondary, 0, 2, vertexBuffersArray.data(), offsets.data());
		vkCmdBindIndexBuffer(secondary, indBuffer, 0, VK_INDEX_TYPE_UINT32);

		uint32_t p = 0;
		for (size_t j = 0; j < objects.size(); j++) {
			uint32_t uniqueModelInd = static_cast<uint32_t>(uniqueModelIndex[objects[j]->meshHash]);
			if (uniqueModelInd == j) { // only process unique models
				// bitfield for which textures exist
				int textureExistence = 0;
				textureExistence |= (objects[j]->material.baseColor.found ? 1 : 0);
				textureExistence |= (objects[j]->material.metallicRoughness.found ? 1 : 0) << 1;
				textureExistence |= (objects[j]->material.normalMap.found ? 1 : 0) << 2;
				textureExistence |= (objects[j]->material.emissiveMap.found ? 1 : 0) << 3;
				textureExistence |= (objects[j]->material.occlusionMap.found ? 1 : 0) << 4;

				struct {
					int textureExist; // bitfield of which textures exist
					int texIndex; // starting index of the textures in the texture array
				} pushConst;

				pushConst.textureExist = textureExistence;
				pushConst.texIndex = meshTexStartInd[p];

				vkCmdPushConstants(secondary, pipe.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConst), &pushConst);
				size_t bufferInd = modelHashToBufferIndex[objects[j]->meshHash];
				uint32_t instanceCount = getModelNumHash(objects[uniqueModelInd]->meshHash);
				vkCmdDrawIndexed(secondary, bufferData[bufferInd].indexCount, instanceCount,
					bufferData[bufferInd].indexOffset, bufferData[bufferInd].vertexOffset, uniqueModelInd);
				p++;
			}
		}

		if (endCommand) {
			if (vkEndCommandBuffer(secondary) != VK_SUCCESS) {
				throw std::runtime_error("failed to record main secondary command buffer!");
			}
		}
	}

	void recordShadowSecondaryCmdBuffers(std::vector<VkCommandBuffer>& secondaries, const PipelineData& pipe, const VkCommandBufferBeginInfo& beginInfo, const VkDescriptorSet* descriptorsets, const size_t descriptorCount, const bool startCommand, const bool endCommand) {
		size_t size = secondaries.size();

		const std::array<VkBuffer, 2> vertexBuffersArray = { vertBuffer, instanceBuffer };
		const std::array<VkDeviceSize, 2> offsets = { 0, 0 };

		for (size_t i = 0; i < size; i++) {
			if (startCommand) {
				if (vkBeginCommandBuffer(secondaries[i], &beginInfo) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording shadow secondary command buffer!");
				}
			}

			vkCmdBindPipeline(secondaries[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.graphicsPipeline);
			vkCmdBindDescriptorSets(secondaries[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, static_cast<uint32_t>(descriptorCount), descriptorsets, 0, nullptr);

			int lightIndex = static_cast<int>(i);
			vkCmdPushConstants(secondaries[i], shadowMapPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(lightIndex), &lightIndex);

			vkCmdBindVertexBuffers(secondaries[i], 0, 2, vertexBuffersArray.data(), offsets.data());
			vkCmdBindIndexBuffer(secondaries[i], indBuffer, 0, VK_INDEX_TYPE_UINT32);
		}

		// iterate through all objects that cast shadows
		uint32_t p = 0;
		for (size_t j = 0; j < objects.size(); j++) {
			uint32_t uniqueModelInd = static_cast<uint32_t>(uniqueModelIndex[objects[j]->meshHash]);
			if (uniqueModelInd == j) {
				size_t bufferInd = modelHashToBufferIndex[objects[j]->meshHash];
				uint32_t instanceCount = getModelNumHash(objects[uniqueModelInd]->meshHash);
				for (size_t i = 0; i < size; i++) {
					vkCmdDrawIndexed(secondaries[i], bufferData[bufferInd].indexCount, instanceCount,
						bufferData[bufferInd].indexOffset, bufferData[bufferInd].vertexOffset, uniqueModelInd);
				}
				p++;
			}
		}

		if (endCommand) {
			for (size_t i = 0; i < size; i++) {
				if (vkEndCommandBuffer(secondaries[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to record shadow secondary command buffer!");
				}
			}
		}
	}

	void recordCompSecondaryCmdBuffers(VkCommandBuffer& secondary, const VkCommandBufferBeginInfo& beginInfo, const VkDescriptorSet* descriptorsets, const size_t descriptorCount, const bool startCommand, const bool endCommand) {
		if (startCommand) {
			if (vkBeginCommandBuffer(secondary, &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording composition secondary command buffer!");
			}
		}

		vkCmdBindPipeline(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, compPipelineData.graphicsPipeline);
		vkCmdBindDescriptorSets(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, compPipelineData.layout, 0, static_cast<uint32_t>(descriptorCount), descriptorsets, 0, nullptr);

		vkCmdDraw(secondary, 6, 1, 0, 0);

		// prepare for next frame in ImGui
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// draw the imgui text
		std::string fpsText = "fps: " + std::to_string(fps);
		std::string objText = "objects: " + std::to_string(objects.size());
		std::string lightText = "lights: " + std::to_string(lights.size());

		ImVec4 bgColor = ImVec4(40.0f, 61.0f, 59.0f, 0.9f);
		drawText(fpsText, static_cast<float>(swap.extent.width / 2), 30, font_large, bgColor);

		float w = ImGui::CalcTextSize(fpsText.c_str()).x + 20;
		float x = static_cast<float>(swap.extent.width / 2) + w;
		drawText(objText, x, 30, font_large, bgColor);

		float w2 = ImGui::CalcTextSize(lightText.c_str()).x + 20;
		float x2 = static_cast<float>(swap.extent.width / 2) - w2;
		drawText(lightText, x2, 30, font_large, bgColor);

		// render the imgui frame and draw imgui's commands into the command buffer
		ImGui::Render();
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), secondary);

		if (endCommand) {
			if (vkEndCommandBuffer(secondary) != VK_SUCCESS) {
				throw std::runtime_error("failed to record composition secondary command buffer!");
			}
		}
	}


	void recordShadowCommandBuffers() {
		const std::array<VkDescriptorSet, 1> shadowDS = { descs.sets[1] };

		VkCommandBufferInheritanceInfo inheritInfo{};
		inheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		inheritInfo.renderPass = shadowMapPipeline.renderPass;
		inheritInfo.framebuffer = VK_NULL_HANDLE;
		inheritInfo.subpass = 0;

		VkCommandBufferBeginInfo beginInfoS{};
		beginInfoS.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfoS.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
		beginInfoS.pInheritanceInfo = &inheritInfo;

		recordShadowSecondaryCmdBuffers(shadowMapCommandBuffers.secondary.buffers, shadowMapPipeline, beginInfoS, shadowDS.data(), 1, true, true);

		VkCommandBufferBeginInfo beginInfoP{};
		beginInfoP.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfoP.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfoP.pInheritanceInfo = nullptr;

		VkClearValue clearValue = { 1.0f, 0 };

		for (size_t i = 0; i < lights.size(); i++) {
#ifdef PROFILE_COMMAND_BUFFERS
#else
			tfCmd.emplace([&, i, beginInfoP, clearValue]() {
#endif
				VkCommandBuffer& shadowCommandBuffer = shadowMapCommandBuffers.primary.buffers[i];
				VkCommandBuffer& secondary = shadowMapCommandBuffers.secondary.buffers[i];

				if (vkBeginCommandBuffer(shadowCommandBuffer, &beginInfoP) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				// render pass
				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = shadowMapPipeline.renderPass;
				renderPassInfo.framebuffer = lights[i]->frameBuffer;
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = { shadowProps.mapWidth, shadowProps.mapHeight };
				renderPassInfo.clearValueCount = 1;
				renderPassInfo.pClearValues = &clearValue;
				vkCmdBeginRenderPass(shadowCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

				vkCmdExecuteCommands(shadowCommandBuffer, 1, &secondary);

				// end the render pass and command buffer
				vkCmdEndRenderPass(shadowCommandBuffer);
				if (vkEndCommandBuffer(shadowCommandBuffer) != VK_SUCCESS) {
					throw std::runtime_error("failed to record command buffer!");
				}
#ifdef PROFILE_COMMAND_BUFFERS
#else
				});
#endif
		}
	}

	void recordMainCommandBuffers() { //records and submits the main command buffers
		const std::array<VkClearValue, 2> clearValues = { VkClearValue{0.18f, 0.3f, 0.30f, 1.0f}, VkClearValue{1.0f, 0} };
		const std::array<VkDescriptorSet, 2> skyboxDS = { descs.sets[3], descs.sets[4] };
		const std::array<VkDescriptorSet, 4> mainDS = { descs.sets[0], descs.sets[1], descs.sets[2], descs.sets[4] };

		const VkDeviceSize skyboxOffset = 0;

		VkCommandBufferInheritanceInfo inheritInfo{};
		inheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		inheritInfo.renderPass = mainPassPipeline.renderPass;
		inheritInfo.framebuffer = VK_NULL_HANDLE;
		inheritInfo.subpass = 0;

		VkCommandBufferBeginInfo beginInfoS{};
		beginInfoS.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfoS.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
		beginInfoS.pInheritanceInfo = &inheritInfo;

		VkCommandBuffer& secondary = mainPassCommandBuffers.secondary[0];
		if (vkBeginCommandBuffer(secondary, &beginInfoS) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording secondary command buffer!");
		}

		// FOR THE SKYBOX PASS
		vkCmdBindPipeline(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipeline);
		vkCmdBindDescriptorSets(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipelineLayout, 0, static_cast<uint32_t>(skyboxDS.size()), skyboxDS.data(), 0, nullptr);
		vkCmdBindVertexBuffers(secondary, 0, 1, &skybox.vertBuffer, &skyboxOffset);
		vkCmdBindIndexBuffer(secondary, skybox.indBuffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(secondary, skybox.bufferData.indexCount, 1, skybox.bufferData.indexOffset, skybox.bufferData.vertexOffset, 0);

		// FOR THE MAIN PASS
		recordMainSecondaryCmdBuffers(secondary, mainPassPipeline, beginInfoS, mainDS.data(), mainDS.size(), false, true);

		VkCommandBufferBeginInfo beginInfoP{};
		beginInfoP.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfoP.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfoP.pInheritanceInfo = nullptr;
		for (size_t i = 0; i < swap.images.size(); i++) {
#ifdef PROFILE_COMMAND_BUFFERS
#else
			tfCmd.emplace([&, i, clearValues, beginInfoP]() {
#endif
				VkCommandBuffer& mainCommandBuffer = mainPassCommandBuffers.primary[i];
				if (vkBeginCommandBuffer(mainCommandBuffer, &beginInfoP) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = mainPassPipeline.renderPass;
				renderPassInfo.framebuffer = mainPassFB;
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = swap.extent;
				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				vkCmdBeginRenderPass(mainCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
				vkCmdExecuteCommands(mainCommandBuffer, static_cast<uint32_t>(mainPassCommandBuffers.secondary.size()), mainPassCommandBuffers.secondary.data());
				vkCmdEndRenderPass(mainCommandBuffer);
				if (vkEndCommandBuffer(mainCommandBuffer) != VK_SUCCESS) {
					throw std::runtime_error("failed to record command buffer!");
				}
#ifdef PROFILE_COMMAND_BUFFERS
#else
				});
#endif
		}
	}

	void recordWBOITCommandBuffers() {
		std::array<VkClearValue, 3> clearValues = { VkClearValue{0.0f, 0.0f, 0.0f, 1.0f}, VkClearValue{1.0f}, VkClearValue{1.0f, 0} };
		std::array<VkDescriptorSet, 5> wboitDS = { descs.sets[0], descs.sets[1], descs.sets[2], descs.sets[4], descs.sets[6] };
		VkDeviceSize offset[] = { 0, 0 };

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		VkCommandBufferInheritanceInfo inheritInfo{};
		inheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		inheritInfo.renderPass = wboitPipeline.renderPass;
		inheritInfo.framebuffer = VK_NULL_HANDLE;
		inheritInfo.subpass = 0;

		VkCommandBufferBeginInfo beginInfoS{};
		beginInfoS.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfoS.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
		beginInfoS.pInheritanceInfo = &inheritInfo;

		recordMainSecondaryCmdBuffers(wboitCommandBuffers.secondary[0], wboitPipeline, beginInfoS, wboitDS.data(), wboitDS.size(), true, true);

		for (size_t i = 0; i < swap.images.size(); i++) {
#ifdef PROFILE_COMMAND_BUFFERS
#else
			tfCmd.emplace([&, i, clearValues, wboitDS, offset, beginInfo]() {
#endif
				VkCommandBuffer& wboitCommandBuffer = wboitCommandBuffers.primary[i];
				if (vkBeginCommandBuffer(wboitCommandBuffer, &beginInfo) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = wboitPipeline.renderPass;
				renderPassInfo.framebuffer = wboit.frameBuffer;
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = swap.extent;
				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				vkhelper::transitionImageLayout(wboitCommandBuffer, mainPassTextures.depth.image, depthFormat, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, 0);

				vkCmdBeginRenderPass(wboitCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
				vkCmdExecuteCommands(wboitCommandBuffer, static_cast<uint32_t>(wboitCommandBuffers.secondary.size()), wboitCommandBuffers.secondary.data());
				vkCmdEndRenderPass(wboitCommandBuffer);

				if (vkEndCommandBuffer(wboitCommandBuffer) != VK_SUCCESS) {
					throw std::runtime_error("failed to record command buffer!");
				}
#ifdef PROFILE_COMMAND_BUFFERS
#else
				});
#endif
		}
	}

	void recordCompCommandBuffers() {
		std::array<VkClearValue, 2> clearValues = { VkClearValue{0.18f, 0.3f, 0.30f, 1.0f}, VkClearValue{1.0f, 0} };
		VkDescriptorSet compDS = descs.sets[5];

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		VkCommandBufferInheritanceInfo inheritInfo{};
		inheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		inheritInfo.renderPass = compPipelineData.renderPass;
		inheritInfo.framebuffer = VK_NULL_HANDLE;
		inheritInfo.subpass = 0;

		VkCommandBufferBeginInfo beginInfoS{};
		beginInfoS.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfoS.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
		beginInfoS.pInheritanceInfo = &inheritInfo;

		recordCompSecondaryCmdBuffers(compCommandBuffers.secondary[0], beginInfoS, &compDS, 1, true, true);

		for (size_t i = 0; i < swap.images.size(); i++) {
#ifdef PROFILE_COMMAND_BUFFERS
#else
			tfCmd.emplace([&, i, clearValues, compDS, beginInfo]() {
#endif
				VkCommandBuffer& compCommandBuffer = compCommandBuffers.primary.buffers[i];
				if (vkBeginCommandBuffer(compCommandBuffer, &beginInfo) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = compPipelineData.renderPass;
				renderPassInfo.framebuffer = swap.framebuffers[i];
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = swap.extent;
				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				vkCmdBeginRenderPass(compCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
				vkCmdExecuteCommands(compCommandBuffer, 1, &compCommandBuffers.secondary[0]);
				vkCmdEndRenderPass(compCommandBuffer);
				if (vkEndCommandBuffer(compCommandBuffer) != VK_SUCCESS) {
					throw std::runtime_error("failed to record command buffer!");
				}
#ifdef PROFILE_COMMAND_BUFFERS
#else
				});
#endif
		}
	}

	void drawText(std::string text, float x, float y, ImFont* font = nullptr, ImVec4 bgColor = ImVec4(-1, -1, -1, -1)) {
		// set the pos and size of ther window
		ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Always);

		// normalize the colors
		bgColor.x /= 255;
		bgColor.y /= 255;
		bgColor.z /= 255;

		if (bgColor.x != -1) {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, bgColor);
		}
		// unique window name
		std::string name = ("window: " + std::to_string(x) + " " + std::to_string(y));

		// get the settings for the window
		ImGui::Begin(
			name.c_str(),
			nullptr,
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoScrollbar |
			ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoInputs |
			ImGuiWindowFlags_NoBringToFrontOnFocus);

		// if font exists, use it. otherwise, use the default font
		if (font != nullptr) {
			ImGui::PushFont(font);
		}
		if (font != nullptr) {
			ImGui::PopFont();
		}
		const char* textChar = text.c_str();

		// center the text
		float fontSize = ImGui::GetFontSize();
		float textWidth = ImGui::CalcTextSize(textChar).x;
		float windowWidth = ImGui::GetWindowSize().x;
		float centeredStartPos = (windowWidth - textWidth) / 2.0f;

		// center the text around the x position cords
		ImGui::SetCursorPosX(centeredStartPos);
		ImGui::TextUnformatted(textChar);

		if (bgColor.x != -1) {
			ImGui::PopStyleColor();  // revert background color change
		}
		ImGui::End();
	}

	void createFrameBuffers(bool initial) {
		if (initial) {
			// create the shadowmap framebuffers
			for (size_t i = 0; i < lights.size(); i++) {
				vkhelper::createFB(shadowMapPipeline.renderPass, lights[i]->frameBuffer, &lights[i]->shadowMapData.imageView, 1, shadowProps.mapWidth, shadowProps.mapHeight);
			}
		}

		// create the main pass framebuffer
		std::vector<VkImageView> attachmentsM = { mainPassTextures.color.imageView, mainPassTextures.depth.imageView };
		vkhelper::createFB(mainPassPipeline.renderPass, mainPassFB, attachmentsM.data(), attachmentsM.size(), swap.extent.width, swap.extent.height);

		// create the wboit framebuffer
		std::vector<VkImageView> attachmentsW = { wboit.weightedColor.imageView };
		vkhelper::createFB(wboitPipeline.renderPass, wboit.frameBuffer, attachmentsW.data(), attachmentsW.size(), swap.extent.width, swap.extent.height);

		// create the composition framebuffers
		size_t swapSize = swap.imageViews.size();
		if (initial) swap.framebuffers.resize(swapSize);
		for (size_t i = 0; i < swapSize; i++) {
			vkhelper::createFB(compPipelineData.renderPass, swap.framebuffers[i], &swap.imageViews[i], 1, swap.extent.width, swap.extent.height);
		}
	}

	void createSemaphores() {
		vkhelper::createSemaphore(imageAvailableSemaphore);
		vkhelper::createSemaphore(renderFinishedSemaphore);
		vkhelper::createSemaphore(shadowSemaphore);
		vkhelper::createSemaphore(wboitSemaphore);
		vkhelper::createSemaphore(compSemaphore);

	}
	void freeTexture(dvl::Texture& t) {
		vkDestroyImageView(device, t.imageView, nullptr);
		vkDestroySampler(device, t.sampler, nullptr);
		vkDestroyImage(device, t.image, nullptr);
		vkFreeMemory(device, t.memory, nullptr);
	}

	void recreateSwap() {
		std::cout << "Recreating swap chain..." << std::endl;
		int width = 0, height = 0;
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		vkDeviceWaitIdle(device); // Wait for device idle

		// free the textures
		freeTexture(mainPassTextures.color);
		freeTexture(mainPassTextures.depth);
		freeTexture(wboit.weightedColor);

		for (auto imageView : swap.imageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swap.swapChain, nullptr);

		createSC();
		createSCImageViews();
		setupTextures();

		for (VkDescriptorPool& pool : descs.pools) {
			vkDestroyDescriptorPool(device, pool, nullptr);
		}
		for (VkDescriptorSetLayout& layout : descs.layouts) {
			vkDestroyDescriptorSetLayout(device, layout, nullptr);
		}

		// create the descriptorsets
		setupDescriptorSets(false);

		// create the pipelines
		setupPipelines(false);

		// create the framebuffers
		createFrameBuffers(false);

		initializeMouseInput(true);
	}

	void drawFrame() { // draw frame function
		uint32_t imageIndex;
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// acquire an image from the swap chain
		VkResult result = vkAcquireNextImageKHR(device, swap.swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			vkDeviceWaitIdle(device);
			recreateSwap();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // stage to wait: color attachment output stage
		std::vector<VkSubmitInfo> submitInfos;

		for (VkCommandBuffer& shadow : shadowMapCommandBuffers.primary.buffers) {
			VkSubmitInfo sub = vkhelper::createSubmitInfo(&shadow, 1);
			submitInfos.push_back(sub);
		}

		submitInfos.push_back(vkhelper::createSubmitInfo(&mainPassCommandBuffers.primary[imageIndex], 1, waitStages, imageAvailableSemaphore, wboitSemaphore));
		submitInfos.push_back(vkhelper::createSubmitInfo(&wboitCommandBuffers.primary[imageIndex], 1, waitStages, wboitSemaphore, compSemaphore));
		submitInfos.push_back(vkhelper::createSubmitInfo(&compCommandBuffers.primary[imageIndex], 1, waitStages, compSemaphore, renderFinishedSemaphore));

		// submit all command buffers in a single call
		if (vkQueueSubmit(graphicsQueue, static_cast<uint32_t>(submitInfos.size()), submitInfos.data(), inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit command buffers!");
		}

		// present the image
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
		VkSwapchainKHR swapChains[] = { swap.swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		//check if the swap chain is out of date (window was resized, etc):
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			vkDeviceWaitIdle(device);
			recreateSwap();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}
	}

	void recordAllCommandBuffers() { // record every command buffer
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

#ifdef PROFILE_COMMAND_BUFFERS
		utils::sep();

		auto now = utils::now();
		recordShadowCommandBuffers();
		auto duration = utils::duration<microseconds>(now);
		std::cout << "recordShadowCommandBuffers: " << utils::durationString(duration) << std::endl;

		now = utils::now();
		recordMainCommandBuffers();
		duration = utils::duration<microseconds>(now);
		std::cout << "recordCommandBuffers: " << utils::durationString(duration) << std::endl;

		now = utils::now();
		recordWBOITCommandBuffers();
		duration = utils::duration<microseconds>(now);
		std::cout << "recordWBOITCommandBuffers: " << utils::durationString(duration) << std::endl;

		now = utils::now();
		recordCompCommandBuffers();
		duration = utils::duration<microseconds>(now);
		std::cout << "recordCompCommandBuffers: " << utils::durationString(duration) << std::endl;
#else
		recordShadowCommandBuffers();
		recordMainCommandBuffers();
		recordWBOITCommandBuffers();
		recordCompCommandBuffers();

		cmdExecutor.run(tfCmd).wait();
		tfCmd.clear();
#endif
	}

	void calcFps(auto& start, auto& prev, uint8_t& frameCount) {
		auto endTime = std::chrono::steady_clock::now();
		frameCount++;

		// calculate FPS every 200 ms
		auto timeSincePrevious = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - prev).count();
		if (timeSincePrevious >= 200) {
			fps = static_cast<uint32_t>(1000 * frameCount / timeSincePrevious);
			frameCount = 0;
			prev = endTime;
		}
	}

	void mainLoop() {
		uint8_t frameCount = 0;
		uint8_t swapSize = static_cast<uint8_t>(swap.images.size());
		auto startTime = std::chrono::steady_clock::now();
		auto previousTime = startTime;

		while (!glfwWindowShouldClose(window)) {
#ifdef PROFILE_MAIN_LOOP
			utils::sep();
			auto now = utils::now();
			currentFrame = (currentFrame + 1) % swapSize;
			auto duration = utils::duration<microseconds>(now);
			std::cout << "currentFrame: " << utils::durationString(duration) << std::endl;

			now = utils::now();
			glfwPollEvents();
			duration = utils::duration<microseconds>(now);
			std::cout << "glfwPollEvents: " << utils::durationString(duration) << std::endl;

			now = utils::now();
			drawFrame();
			duration = utils::duration<microseconds>(now);
			std::cout << "drawFrame: " << utils::durationString(duration) << std::endl;

			now = utils::now();
			handleKeyboardInput();
			duration = utils::duration<microseconds>(now);
			std::cout << "handleKeyboardInput: " << utils::durationString(duration) << std::endl;

			now = utils::now();
			recordAllCommandBuffers();
			duration = utils::duration<microseconds>(now);
			std::cout << "recordAllCommandBuffers: " << utils::durationString(duration) << std::endl;

			now = utils::now();
			updateUBO();
			duration = utils::duration<microseconds>(now);
			std::cout << "updateUBO: " << utils::durationString(duration) << std::endl;

			now = utils::now();
			calcFps(startTime, previousTime, frameCount);
			duration = utils::duration<microseconds>(now);
			std::cout << "calcFps: " << utils::durationString(duration) << std::endl;
#else PROFILE_FRAMES
			currentFrame = (currentFrame + 1) % swapSize;
			glfwPollEvents();
			drawFrame();
			handleKeyboardInput(); // handle keyboard input
			recordAllCommandBuffers();
			updateUBO(); // update ubo matrices and populate the buffer
			calcFps(startTime, previousTime, frameCount);
#endif
		}

		vkDeviceWaitIdle(device);
	}

	void initializeMouseInput(bool initial) {
		// set the lastX and lastY to the center of the screen
		if (initial) {
			cam.lastX = static_cast<float>(swap.extent.width) / 2.0f;
			cam.lastY = static_cast<float>(swap.extent.height) / 2.0f;
			glfwSetCursorPos(window, cam.lastX, cam.lastY);
		}

		// only hide and capture cursor if cam.locked is true
		if (cam.locked) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

			// set the mouse callback
			glfwSetCursorPosCallback(window, mouseCallback);
		}
		else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}

	}

	static void mouseCallback(GLFWwindow* window, double xPos, double yPos) {
		static bool mouseFirst = true;
		float xp = static_cast<float>(xPos);
		float yp = static_cast<float>(yPos);

		if (cam.locked) {
			float xoff = cam.lastX - xp;
			float yoff = cam.lastY - yp;
			cam.lastX = xp;
			cam.lastY = yp;

			float sens = 0.1f;
			xoff *= sens;
			yoff *= sens;

			cam.rightAngle += xoff;
			cam.upAngle += yoff;
		}
	}

	void handleKeyboardInput() {
		double currentFrame = glfwGetTime();
		float deltaTime = static_cast<float>(currentFrame - lastFrame);
		lastFrame = currentFrame;

		float cameraSpeed = 2.0f * deltaTime;

		cam.upAngle = fmod(cam.upAngle + 360.0f, 360.0f);
		cam.rightAngle = fmod(cam.rightAngle + 360.0f, 360.0f);

		cam.updateQuaternion();
		dml::vec3 forward = dml::quatToDir(cam.quat);
		dml::vec3 right = dml::normalize(dml::cross(forward, dml::vec3(0, 1, 0)));

		// camera movement
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			cam.camPos -= forward * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			cam.camPos += forward * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			cam.camPos -= right * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			cam.camPos += right * cameraSpeed;
		}

		// up and down movement
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
			cam.camPos.y += 1 * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
			cam.camPos.y -= 1 * cameraSpeed;
		}

		// realtime object loading
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
			summonModel();
		}

		// realtime light loading
		if (eKey.isPressed()) {
			summonLight();
		}

		if (rKey.isPressed()) {
			uint64_t vertCount = 0;
			for (const auto& o : objects) {
				vertCount += o->vertices.size();
			}

			double startS = ((double)vertCount / fps) / 1000;
			double memEfficiency = vertCount / (1024.0 * 1024.0); // convert to mb
			double finalS = 100 * ((startS) / memEfficiency / lights.size());

			utils::sep();
			std::cout << "Vertex count: " << vertCount << std::endl;
			std::cout << "Object count: " << objects.size() << std::endl;
			std::cout << "Light count: " << lights.size() << " / " << MAX_LIGHTS << std::endl;
			std::cout << "Score: " << finalS << std::endl;
		}

		// lock / unlock mouse
		if (escapeKey.isPressed()) {
			cam.locked = !cam.locked;
			initializeMouseInput(cam.locked);
		}
	}

	void initVulkan() { //initializes Vulkan functions
		// initialize Vulkan components
		auto now = utils::now();
		createInstance();
		createSurface();
		pickDevice();
		createLogicalDevice();
		initQueues(); //sets the queue family indices such as graphics and presentation
		createSC(); //create swap chain
		setupFences();
		createSemaphores();

		commandPool = createCommandPool();
		initializeMouseInput(true);
		loadUniqueObjects();

		// create buffers and textures
		createModelBuffers();
		setupTextures();
		loadSkybox("night-sky.hdr");
		setupBuffers();

		// setup the descriptorsets and pipelines
		setupDescriptorSets();
		setupPipelines(true);
		imguiSetup();
		updateUBO();

		// setup the framebuffers and command buffers
		createFrameBuffers(true);
		createCommandBuffers();
		recordAllCommandBuffers();

		auto duration = utils::duration<milliseconds>(now);
		utils::printDuration(duration);

		std::cout << "Vulkan initialized successfully! Unique models: " << getUniqueModels() << std::endl;
	}
};
int main() {
	Engine app;
	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}