#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

//#define PROFILE_MAIN_LOOP
//#define PROFILE_COMMAND_BUFFERS
//#define ENABLE_DEBUG

#include <stb_image_resize.h>

// headers
#include <dml.hpp>
#include <dvl.hpp>
#include <utils.hpp>
#include <vkhelper.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <taskflow.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <optional>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <array>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <random>
#include <ctime>
#include <cmath>
#include <memory>

using microseconds = std::chrono::microseconds;
using milliseconds = std::chrono::milliseconds;

constexpr uint32_t MAX_MODELS = 1200;
constexpr uint32_t MAX_LIGHTS = 300;

constexpr uint32_t SCREEN_WIDTH = 3200;
constexpr uint32_t SCREEN_HEIGHT = 1800;

const std::string ENGINE_VER = "v0.5.1";

const std::string SHADER_DIR = "shaders/compiled/";
const std::string MODEL_DIR = "assets/models/";
const std::string SKYBOX_DIR = "assets/skyboxes/";
const std::string FONT_DIR = "assets/fonts/";

bool rtSupported = false; // a bool if raytracing is supported on the device
bool rtEnabled = true; // a bool if raytracing has been enabled

struct MouseData {
	bool locked;

	float lastX;
	float lastY;

	float upAngle;
	float rightAngle;

	MouseData()
		: upAngle(0.0f),
		rightAngle(0.0f),
		lastX(0.0f),
		lastY(0.0f),
		locked(true)
	{}
};

// globals
MouseData mouse;
VkQueue graphicsQueue{};
VkPhysicalDevice physicalDevice{};
GLFWwindow* window = nullptr;

VkSurfaceKHR surface{};
VkInstance instance{};
VkDevice device{};

class Engine {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
	}
private:
	struct CamData {
		dml::vec3 pos; //x, y, z
		dml::vec4 quat;

		dml::mat4 projectionMatrix;
		dml::mat4 viewMatrix;

		// buffers for the camera matrix ubo
		VkhBuffer buffer;
		VkhDeviceMemory bufferMem;

		float fov;
		float nearP;
		float farP;

		CamData()
			: pos(0.0f, 0.0f, 0.0f),
			quat(0.0f, 0.0f, 0.0f, 1.0f),
			projectionMatrix(),
			viewMatrix(),
			buffer(),
			bufferMem(),
			fov(60.0f),
			nearP(0.01f),
			farP(100.0f)
		{}

		const dml::mat4 getViewMatrix(MouseData& m) {
			return dml::viewMatrix(pos, dml::radians(m.upAngle), dml::radians(m.rightAngle));
		}

		void updateQuaternion(MouseData& m) {
			dml::vec4 yRot = dml::angleAxis(dml::radians(m.upAngle), dml::vec3(1, 0, 0));
			dml::vec4 xRot = dml::angleAxis(dml::radians(m.rightAngle), dml::vec3(0, 1, 0));
			quat = yRot * xRot;
		}
	};

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
		VkhFramebuffer frameBuffer;
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
			frameBuffer(),
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
		uint32_t width = 2048;
		uint32_t height = 2048;
	};

	struct SkyboxObject { // skybox struct
		dvl::Texture cubemap;
		VkhPipelineLayout pipelineLayout;
		VkhPipeline pipeline;
		vkh::BufData bufferData; // buffer data for the skybox (vert offsets, etc)
		VkhBuffer vertBuffer;
		VkhDeviceMemory vertBufferMem;
		VkhBuffer indBuffer;
		VkhDeviceMemory indBufferMem;

		std::vector<dml::vec3> vertices;
		std::vector<uint32_t> indices;

		void resetPipeline() {
			if (pipelineLayout.valid()) pipelineLayout.reset();
			if (pipeline.valid()) pipeline.reset();
		}

		SkyboxObject()
			: cubemap(),
			pipelineLayout(),
			pipeline(),
			bufferData(),
			vertBuffer(),
			vertBufferMem(),
			indBuffer(),
			indBufferMem(),

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
		std::vector<VkhDescriptorPool> pools;
		std::vector<VkhDescriptorSetLayout> layouts;
		std::vector<VkhDescriptorSet> sets;
	};

	struct PipelineData {
		VkhRenderPass renderPass;
		VkhPipelineLayout layout;
		VkhPipeline pipeline;

		void reset() {
			if (renderPass.valid()) renderPass.reset();
			if (layout.valid()) layout.reset();
			if (pipeline.valid()) pipeline.reset();
		}
	};

	struct SCData {
		VkhSwapchainKHR swapChain;
		std::vector<VkhImage> images;
		VkFormat imageFormat;
		VkExtent2D extent;
		std::vector<VkhImageView> imageViews;
		uint32_t imageCount;
		std::vector<VkhFramebuffer> framebuffers;

		SCData()
			: swapChain(),
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
		VkhFramebuffer frameBuffer;

		WBOITData()
			: weightedColor(),
			frameBuffer()
		{}
	};

	struct OpaquePassTex {
		dvl::Texture depth;
		dvl::Texture color;
	};

	struct CommandBufferCollection {
		std::vector<VkhCommandPool> pools;

		std::vector<VkhCommandBuffer> buffers;
		std::vector<VkCommandBuffer> rawBuffers;

		void reserveClear(size_t size) {
			buffers.clear();
			pools.clear();

			buffers.reserve(size);
			pools.reserve(size);
		}

		size_t size() const {
			return buffers.size();
		}

		VkCommandBuffer* data() {
			if (buffers.size() == 1) {
				return buffers[0].p();
			}
			else {
				rawBuffers.clear();
				rawBuffers.reserve(buffers.size());
				for (const auto& b : buffers) {
					rawBuffers.push_back(b.v());
				}
				return rawBuffers.data();
			}
		}

		VkhCommandBuffer& operator[](size_t index) {
			return buffers[index];
		}
	};

	struct CommandBufferSet {
		CommandBufferCollection primary;
		CommandBufferCollection secondary;
	};

	struct BlasData {
		VkhAccelerationStructure blas;
		VkhBuffer compBuffer;
		VkhDeviceMemory compMem;
	};


	CamData cam;

	// window and rendering context
	VkQueue presentQueue = VK_NULL_HANDLE;
	VkQueue computeQueue = VK_NULL_HANDLE;
	VkQueue transferQueue = VK_NULL_HANDLE;
	vkh::QueueFamilyIndices queueFamilyIndices;

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
	PipelineData opaquePassPipeline;
	PipelineData shadowMapPipeline;
	PipelineData compPipelineData;
	PipelineData wboitPipeline;

	VkhFramebuffer opaquePassFB;
	OpaquePassTex opaquePassTextures = {};

	// command buffers and command pool
	VkhCommandPool commandPool;
	CommandBufferSet opaquePassCommandBuffers;
	CommandBufferSet shadowMapCommandBuffers;
	CommandBufferSet wboitCommandBuffers;
	CommandBufferSet compCommandBuffers;

	// buffers
	VkhBuffer vertBuffer;
	VkhBuffer indBuffer;
	VkhBuffer instanceBuffer;
	VkhBuffer lightBuffer;
	VkhBuffer sceneIndexBuffer;

	// buffer memory
	VkhDeviceMemory vertBufferMem;
	VkhDeviceMemory indBufferMem;
	VkhDeviceMemory instanceBufferMem;
	VkhDeviceMemory lightBufferMem;
	VkhDeviceMemory sceneIndexBufferMem;

	// synchronization primitives
	std::vector<VkhFence> inFlightFences;
	VkhSemaphore imageAvailableSemaphore;
	VkhSemaphore renderFinishedSemaphore;
	VkhSemaphore shadowSemaphore;
	VkhSemaphore wboitSemaphore;
	VkhSemaphore compSemaphore;

	// descriptor sets and pools
	DSObject descs = {};
	VkhDescriptorPool imguiDescriptorPool;
	VkhDescriptorSetLayout imguiDescriptorSetLayout;

	// vulkan function pointers

	// scene data and objects
	std::vector<vkh::BufData> bufferData;
	std::vector<std::unique_ptr<dvl::Mesh>> objects;
	std::vector<std::unique_ptr<dvl::Mesh>> originalObjects;
	std::vector<uint32_t> playerModels;
	ModelMatInstanceData objInstanceData = {};
	CamUBO camMatData = {};
	LightDataSSBO lightData = {};
	std::vector<std::unique_ptr<Light>> lights;

	// path tracing
	std::vector<BlasData> BLAS;

	VkhAccelerationStructure TLAS;
	VkhBuffer tlasBuffer;
	VkhDeviceMemory tlasMem;
	VkhBuffer tlasInstanceBuffer;
	VkhDeviceMemory tlasInstanceBufferMem;
	VkhBuffer tlasScratchBuffer;
	VkhDeviceMemory tlasScratchMem;
	VkhBuffer compTlasBuffer;
	VkhDeviceMemory compTlasMem;

	std::vector<VkAccelerationStructureInstanceKHR> meshInstances;

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
	dvl::Texture compTex = dvl::Texture(VK_SAMPLE_COUNT_8_BIT);
	VkFormat depthFormat = VK_FORMAT_UNDEFINED;
	WBOITData wboit = {};

	// skybox data
	float* skyboxData = nullptr;
	SkyboxObject skybox = {};

	// font data
	ImFont* font_small = nullptr;
	ImFont* font_large = nullptr;

	// performance metrics
	uint32_t fps = 0;
	double lastFrame = 0.0;

	// mutexes and taskflow
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

		std::string engineName = "3d-vulkan " + ENGINE_VER;

		window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, engineName.c_str(), nullptr, nullptr);
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForVulkan(window, true);

		font_large = ImGui::GetIO().Fonts->AddFontFromFileTTF((FONT_DIR + "OpenSans/OpenSans-VariableFont_wdth,wght.ttf").c_str(), 50.0f);
	}

	void validateFiles() {
		// verify that the assets/models/ directory exists. if not, make it
		std::string models = "assets/models/";

		if (!std::filesystem::exists(models)) {
			if (std::filesystem::create_directory(models)) {
#ifdef ENABLE_DEBUG
				std::cout << "Created directory: " << models << std::endl;
#endif
			}
			else {
				throw std::runtime_error("Failed to create directory" + models + "!");
			}
		}
	}

	void createObject(std::string path, dml::vec3 scale, dml::vec4 rotation, dml::vec3 pos) {
		loadModel(scale, pos, rotation, MODEL_DIR + path);
	}

	Light createLight(dml::vec3 pos, dml::vec3 t, dml::vec3 color = { 1.0f, 1.0f, 1.0f }, float intensity = 2.0f) {
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

	Light createPlayerLight(dml::vec3 color = { 1.0f, 1.0f, 1.0f }, float intensity = 2.0f) {
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
		createObjTask(tfMesh, "sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQuat({ 3.0f, 1.0f, -2.11f }, { 0.0f, 0.0f, 0.0f }), { 3.0f, 1.0f, -2.11f });
		createObjTask(tfMesh, "sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQuat({ -2.0f, 0.0f, 2.11f }, { 0.0f, 0.0f, 0.0f }), { -2.0f, 0.0f, 2.11f });
		createObjTask(tfMesh, "sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQuat({ 0.0f, 2.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }), { 0.0f, 2.0f, 0.0f });

		meshExecutor.run(tfMesh).wait();

		if (!objects.size()) {
			throw std::runtime_error("Failed to load models!");
		}

		lights.push_back(std::make_unique<Light>(createLight({ -2.0f, 0.0f, -4.0f }, { 0.0f, 1.4f, 0.0f })));
		lights.push_back(std::make_unique<Light>(createLight({ -2.0f, 0.0f, 4.0f }, { 0.0f, 1.4f, 0.0f })));
		lights.push_back(std::make_unique<Light>(createPlayerLight()));

		for (auto& obj : objects) {
			originalObjects.push_back(std::make_unique<dvl::Mesh>(*obj));
		}

		// setPlayer(6);
		// setPlayer(9);
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

		// check if the device supports acceleration structure features
		VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
		asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		asFeatures.pNext = &rtFeatures;

		VkPhysicalDeviceFeatures2 deviceFeatures2{};
		deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		deviceFeatures2.pNext = &asFeatures;

		vkGetPhysicalDeviceFeatures2(device, &deviceFeatures2);

		return (rtFeatures.rayTracingPipeline == VK_TRUE && asFeatures.accelerationStructure == VK_TRUE);
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
			vkh::QueueFamilyIndices indices = vkh::findQueueFamilyIndices(surface, device);
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
		queueFamilyIndices = vkh::findQueueFamilyIndices(surface, bestDevice);

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
		rtEnabled = rtEnabled && rtSupported;
		std::cout << "Raytacing is " << (rtEnabled ? "enabled" : "not enabled") << "!" << std::endl;
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

	void createLogicalDevice() {
		float queuePriority = 1.0f;
		VkDeviceQueueCreateInfo queueInf{};
		queueInf.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueInf.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); // index of the graphics queue family
		queueInf.queueCount = 1;
		queueInf.pQueuePriorities = &queuePriority;

		VkPhysicalDeviceNestedCommandBufferFeaturesEXT nestedCommandBufferFeatures{};
		nestedCommandBufferFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES_EXT;
		nestedCommandBufferFeatures.nestedCommandBufferSimultaneousUse = VK_TRUE;
		nestedCommandBufferFeatures.nestedCommandBuffer = VK_TRUE;
		nestedCommandBufferFeatures.nestedCommandBufferRendering = VK_TRUE;

		VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures = {};
		bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
		bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
		bufferDeviceAddressFeatures.pNext = &nestedCommandBufferFeatures;

		VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
		VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};

		if (rtEnabled) {
			accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
			accelerationStructureFeatures.accelerationStructure = VK_TRUE;
			accelerationStructureFeatures.pNext = &bufferDeviceAddressFeatures;

			rayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
			rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
			rayTracingPipelineFeatures.pNext = &accelerationStructureFeatures;
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.imageCubeArray = VK_TRUE;
		deviceFeatures.sampleRateShading = VK_TRUE;

		VkPhysicalDeviceDescriptorIndexingFeatures descIndexing{};
		descIndexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
		descIndexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
		descIndexing.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
		descIndexing.runtimeDescriptorArray = VK_TRUE;
		descIndexing.descriptorBindingVariableDescriptorCount = VK_TRUE;
		descIndexing.descriptorBindingPartiallyBound = VK_TRUE;

		if (rtEnabled) descIndexing.pNext = &rayTracingPipelineFeatures;
		else descIndexing.pNext = &bufferDeviceAddressFeatures;

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
			VK_EXT_NESTED_COMMAND_BUFFER_EXTENSION_NAME,
			VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
		};

		if (rtEnabled) {
			deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
		}

		vkhfp::loadFuncPointers(instance);

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
			throw std::runtime_error("Failed to create logical device!");
		}

		utils::sep();
	}

	std::vector<char> readFile(const std::string& filename) { //outputs binary data from a SPIRV shader file
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

	void createSC() {
		vkh::SCsupportDetails swapChainSupport = vkh::querySCsupport(surface);

		// choose the best surface format, present mode, and swap extent for the swap chain
		VkSurfaceFormatKHR surfaceFormat = vkh::chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR present = vkh::chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = vkh::chooseSwapExtent(swapChainSupport.capabilities, SCREEN_WIDTH, SCREEN_HEIGHT);

		// get the number of images for the sc. this is the minumum + 1
		swap.imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0 && swap.imageCount > swapChainSupport.capabilities.maxImageCount) {
			swap.imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		// get the graphics queue family indices
		uint32_t graphicsIndices[] = { queueFamilyIndices.graphicsFamily.value() };

		// create the swap chain
		VkSwapchainCreateInfoKHR newinfo{};
		newinfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		newinfo.surface = surface;
		newinfo.minImageCount = swap.imageCount;
		newinfo.imageFormat = surfaceFormat.format;
		newinfo.imageColorSpace = surfaceFormat.colorSpace;
		newinfo.imageExtent = extent;
		newinfo.imageArrayLayers = 1;
		newinfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		// allows only 1 queue family to access the sc at a time
		// this reduces synchronization overhead
		newinfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		newinfo.queueFamilyIndexCount = 1;
		newinfo.pQueueFamilyIndices = graphicsIndices; // which queue families will handle the swap chain images
		newinfo.preTransform = swapChainSupport.capabilities.currentTransform; // transform to apply to the swap chain before presentation
		newinfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // set the alpha channel to opaque when compositing the final image
		newinfo.presentMode = present;
		newinfo.clipped = VK_TRUE; // if the window is obscured, the pixels that are obscured will not be drawn to
		newinfo.oldSwapchain = VK_NULL_HANDLE;
		if (vkCreateSwapchainKHR(device, &newinfo, nullptr, swap.swapChain.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		// get the swap chain images
		vkGetSwapchainImagesKHR(device, swap.swapChain.v(), &swap.imageCount, nullptr);

		swap.images.resize(swap.imageCount);
		std::vector<VkImage> images(swap.imageCount);

		vkGetSwapchainImagesKHR(device, swap.swapChain.v(), &swap.imageCount, images.data());
		for (uint32_t i = 0; i < swap.imageCount; i++) {
			swap.images[i] = VkhImage(images[i]);
			swap.images[i].setDestroy(false); // obj wont be automatically freed when out of scope
		}

		swap.imageFormat = surfaceFormat.format;
		swap.extent = extent;

		createSCImageViews();

		// create the viewport for the swap chain
		swapVP.x = 0.0f;
		swapVP.y = 0.0f;
		swapVP.width = static_cast<float>(swap.extent.width);
		swapVP.height = static_cast<float>(swap.extent.height);
		swapVP.minDepth = 0.0f;
		swapVP.maxDepth = 1.0f;
	}

	void loadSkybox(std::string path) {
		skybox.cubemap.path = SKYBOX_DIR + path;
		createTexturedCubemap(skybox.cubemap);
		vkh::createImageView(skybox.cubemap, vkh::CUBEMAP);
		vkh::createSampler(skybox.cubemap.sampler, skybox.cubemap.mipLevels, vkh::CUBEMAP);

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
			vkh::createImageView(newObject.material.baseColor);
			vkh::createSampler(newObject.material.baseColor.sampler, newObject.material.baseColor.mipLevels);

		}

		if (newObject.material.metallicRoughness.found) {
			createTexturedImage(newObject.material.metallicRoughness, true, vkh::METALLIC);
			vkh::createImageView(newObject.material.metallicRoughness, vkh::METALLIC);
			vkh::createSampler(newObject.material.metallicRoughness.sampler, newObject.material.metallicRoughness.mipLevels);

		}

		if (newObject.material.normalMap.found) {
			createTexturedImage(newObject.material.normalMap, true, vkh::NORMAL);
			vkh::createImageView(newObject.material.normalMap, vkh::NORMAL);
			vkh::createSampler(newObject.material.normalMap.sampler, newObject.material.normalMap.mipLevels);

		}

		if (newObject.material.emissiveMap.found) {
			createTexturedImage(newObject.material.emissiveMap, true, vkh::EMISSIVE);
			vkh::createImageView(newObject.material.emissiveMap, vkh::EMISSIVE);
			vkh::createSampler(newObject.material.emissiveMap.sampler, newObject.material.emissiveMap.mipLevels);

		}

		if (newObject.material.occlusionMap.found) {
			createTexturedImage(newObject.material.occlusionMap, true, vkh::OCCLUSION);
			vkh::createImageView(newObject.material.occlusionMap, vkh::OCCLUSION);
			vkh::createSampler(newObject.material.occlusionMap.sampler, newObject.material.occlusionMap.mipLevels);

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
			std::cerr << err << std::endl;
			return;
		}
		if (!ret) {
			std::cerr << "Failed to load GLTF model" << std::endl;
			return;
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
		depthFormat = vkh::findDepthFormat();
		static bool init = true;

		// opaque pass color image
		vkh::createImage(opaquePassTextures.color.image, opaquePassTextures.color.memory, swap.extent.width, swap.extent.height, swap.imageFormat, 1, 1, false, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			commandPool, opaquePassTextures.color.sampleCount);
		vkh::createImageView(opaquePassTextures.color, swap.imageFormat);
		vkh::createSampler(opaquePassTextures.color.sampler, opaquePassTextures.color.mipLevels);

		// opaque pass depth image
		vkh::createImage(opaquePassTextures.depth.image, opaquePassTextures.depth.memory, swap.extent.width, swap.extent.height, depthFormat, 1, 1, false, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			opaquePassTextures.depth.sampleCount);
		vkh::createImageView(opaquePassTextures.depth, vkh::DEPTH);
		vkh::createSampler(opaquePassTextures.depth.sampler, opaquePassTextures.depth.mipLevels, vkh::DEPTH);

		// weighted color image
		vkh::createImage(wboit.weightedColor.image, wboit.weightedColor.memory, swap.extent.width, swap.extent.height, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, false, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			wboit.weightedColor.sampleCount);
		vkh::createImageView(wboit.weightedColor, VK_FORMAT_R16G16B16A16_SFLOAT);
		vkh::createSampler(wboit.weightedColor.sampler, wboit.weightedColor.mipLevels);

		// composition image
		vkh::createImage(compTex.image, compTex.memory, swap.extent.width, swap.extent.height, swap.imageFormat, 1, 1, false, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			compTex.sampleCount);
		vkh::createImageView(compTex, swap.imageFormat);
		vkh::createSampler(compTex.sampler, compTex.mipLevels);

		// shadowmaps
		if (init) {
			for (size_t i = 0; i < lights.size(); i++) {
				vkh::createImage(lights[i]->shadowMapData.image, lights[i]->shadowMapData.memory, shadowProps.width, shadowProps.height, depthFormat, 1, 1, false, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					lights[i]->shadowMapData.sampleCount);
				vkh::createImageView(lights[i]->shadowMapData, vkh::DEPTH);
				vkh::createSampler(lights[i]->shadowMapData.sampler, lights[i]->shadowMapData.mipLevels, vkh::DEPTH);
			}
			init = false;
		}
	}

	void createSCImageViews() { //create the image views for the swap chain images
		swap.imageViews.resize(swap.images.size());

		for (size_t i = 0; i < swap.images.size(); i++) {
			if (swap.imageViews[i].valid()) swap.imageViews[i].reset();
			VkImageViewCreateInfo newinfo{};
			newinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			newinfo.image = swap.images[i].v(); // assign the current swap chain image
			newinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			newinfo.format = swap.imageFormat;

			// image will maintain its original component ordering
			newinfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			newinfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			newinfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			newinfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			newinfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			newinfo.subresourceRange.baseMipLevel = 0;
			newinfo.subresourceRange.levelCount = 1;
			newinfo.subresourceRange.baseArrayLayer = 0;
			newinfo.subresourceRange.layerCount = 1;
			VkResult result = vkCreateImageView(device, &newinfo, nullptr, swap.imageViews[i].p());
			if (result != VK_SUCCESS) {
				throw std::runtime_error("Failed to create image views for the swapchain!!");
			}
		}
	}

	void createLightBuffer() {
		lightData.lightCords.resize(lights.size());
		lightData.memSize = lights.size() * sizeof(LightDataObject);

		vkh::createBuffer(lightBuffer, lightBufferMem, lightData.lightCords.data(), lightData.memSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, commandPool, graphicsQueue, 0, false);
	}

	void setupBuffers() {
		vkh::createBuffer(instanceBuffer, instanceBufferMem, objInstanceData, sizeof(ModelMatInstanceData), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, commandPool, graphicsQueue, 0, false);
		vkh::createBuffer(cam.buffer, cam.bufferMem, camMatData, sizeof(CamUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, commandPool, graphicsQueue, 0, false);

		createLightBuffer();

		// skybox buffer data
		vkh::createBuffer(skybox.vertBuffer, skybox.vertBufferMem, skybox.vertices.data(), sizeof(dml::vec3) * skybox.bufferData.vertexCount, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, commandPool, graphicsQueue, 0);
		vkh::createBuffer(skybox.indBuffer, skybox.indBufferMem, skybox.indices.data(), sizeof(uint32_t) * skybox.bufferData.indexCount, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, commandPool, graphicsQueue, 0);
	}

	void calcCameraMats() {
		cam.viewMatrix = cam.getViewMatrix(mouse);
		cam.projectionMatrix = dml::projection(cam.fov, swap.extent.width / static_cast<float>(swap.extent.height), cam.nearP, cam.farP);
	}

	void calcShadowMats(Light& l) {
		if (l.followPlayer) {
			l.pos = dml::getCamWorldPos(cam.viewMatrix);
			l.target = l.pos + dml::quatToDir(cam.quat);
		}
		// spotlight shadow mapping math code
		float aspectRatio = static_cast<float>(shadowProps.width) / static_cast<float>(shadowProps.height);
		float nearPlane = 0.01f, farPlane = 100.0f;

		dml::vec3 up = dml::vec3(0.0f, 1.0f, 0.0f);
		if (l.pos == l.target) {
			std::cerr << "Light position and target are the same!" << std::endl;
			return;
		}

		l.view = dml::lookAt(l.pos, l.target, up);
		l.proj = dml::projection(l.outerConeAngle + 15.0f, aspectRatio, nearPlane, farPlane);
	}

	void updateUBO() {
		// calc matricies for lights
		for (size_t i = 0; i < lights.size(); i++) {
			Light& l = *lights[i];
			calcShadowMats(l);
			copyLightToLightCords(l, lightData.lightCords[i]);
		}

		void* lData;
		vkMapMemory(device, lightBufferMem.v(), 0, lightData.memSize, 0, &lData);
		memcpy(lData, lightData.lightCords.data(), lightData.memSize);
		vkUnmapMemory(device, lightBufferMem.v());

		// calc matricies for camera
		calcCameraMats();
		memcpy(&camMatData.view, &cam.viewMatrix, sizeof(cam.viewMatrix));
		memcpy(&camMatData.proj, &cam.projectionMatrix, sizeof(cam.projectionMatrix));

		void* cData;
		vkMapMemory(device, cam.bufferMem.v(), 0, sizeof(camMatData), 0, &cData);
		memcpy(cData, &camMatData, sizeof(camMatData));
		vkUnmapMemory(device, cam.bufferMem.v());

		// calc matricies for objects
		for (size_t i = 0; i < objects.size(); i++) {
			int render = 0;
			if (objects[i]->player) {
				dml::mat4 t = dml::translate(cam.pos);
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
		vkMapMemory(device, instanceBufferMem.v(), 0, sizeof(objInstanceData), 0, &matrixData);
		memcpy(matrixData, &objInstanceData, sizeof(objInstanceData));
		vkUnmapMemory(device, instanceBufferMem.v());
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

	void getAllTextures() {
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
	}

	template<typename Stage>
	void createDSLayoutPool(uint32_t index, VkDescriptorType type, uint32_t size, Stage shaderStage) {
		descs.layouts.emplace_back(vkh::createDSLayout(index, type, size, shaderStage));
		descs.pools.emplace_back(vkh::createDSPool(type, size));
	}

	void createDS() {
		std::vector<VkDescriptorImageInfo> imageInfos(totalTextureCount);
		for (size_t i = 0; i < totalTextureCount; i++) {
			imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfos[i].imageView = allTextures[i].imageView.v();
			imageInfos[i].sampler = allTextures[i].sampler.v();
		}

		uint32_t lightSize = static_cast<uint32_t>(lights.size());
		//uint32_t lightSize = MAX_LIGHTS;

		VkDescriptorBufferInfo lightBufferInfo{};
		lightBufferInfo.buffer = lightBuffer.v();
		lightBufferInfo.offset = 0;
		lightBufferInfo.range = lightData.memSize;

		shadowInfos.resize(lightSize);
		for (size_t i = 0; i < lightSize; i++) {
			shadowInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			shadowInfos[i].imageView = lights[i]->shadowMapData.imageView.v();
			shadowInfos[i].sampler = lights[i]->shadowMapData.sampler.v();
		}

		VkDescriptorImageInfo skyboxInfo{};
		skyboxInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		skyboxInfo.imageView = skybox.cubemap.imageView.v();
		skyboxInfo.sampler = skybox.cubemap.sampler.v();

		VkDescriptorBufferInfo camMatBufferInfo{};
		camMatBufferInfo.buffer = cam.buffer.v();
		camMatBufferInfo.offset = 0;
		camMatBufferInfo.range = sizeof(CamUBO);

		const uint32_t texCompSize = 2;
		std::vector<VkDescriptorImageInfo> compositionPassImageInfo(texCompSize);
		for (size_t i = 0; i < texCompSize; i++) {
			compositionPassImageInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			compositionPassImageInfo[i].imageView = (i == 0) ? opaquePassTextures.color.imageView.v() : wboit.weightedColor.imageView.v();
			compositionPassImageInfo[i].sampler = (i == 0) ? opaquePassTextures.color.sampler.v() : wboit.weightedColor.sampler.v();
		}

		VkDescriptorImageInfo opaquePassDepthInfo{};
		opaquePassDepthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		opaquePassDepthInfo.imageView = opaquePassTextures.depth.imageView.v();
		opaquePassDepthInfo.sampler = opaquePassTextures.depth.sampler.v();

		const uint8_t size = 7;
		descs.sets.clear(); descs.sets.reserve(size);
		descs.pools.clear(); descs.pools.reserve(size);
		descs.layouts.clear(); descs.layouts.reserve(size);

		uint32_t texSize = static_cast<uint32_t>(totalTextureCount);

		//initialize descriptor set layouts and pools
		createDSLayoutPool(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texSize, VK_SHADER_STAGE_FRAGMENT_BIT); // array of textures
		createDSLayoutPool(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)); // light data ssbo
		createDSLayoutPool(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_LIGHTS, VK_SHADER_STAGE_FRAGMENT_BIT); // array of shadow map samplers
		createDSLayoutPool(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // 1 sampler for the skybox
		createDSLayoutPool(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT); // camera matricies ubo
		createDSLayoutPool(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texCompSize, VK_SHADER_STAGE_FRAGMENT_BIT); // textures for composition pass
		createDSLayoutPool(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // texture for opaque pass depth

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorSetCount = 1;

		std::vector<uint32_t> descCountArr = { static_cast<uint32_t>(imageInfos.size()), 1, MAX_LIGHTS, 1, 1, texCompSize, 1 };
		for (uint32_t i = 0; i < size; i++) {
			VkDescriptorSetVariableDescriptorCountAllocateInfoEXT varCountInfo{};
			varCountInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT;
			varCountInfo.descriptorSetCount = 1;
			varCountInfo.pDescriptorCounts = &descCountArr[i];
			allocInfo.pNext = &varCountInfo;

			allocInfo.descriptorPool = descs.pools[i].v();
			allocInfo.pSetLayouts = descs.layouts[i].p();
			VkhDescriptorSet set(descs.pools[i].v());
			VkResult result = vkAllocateDescriptorSets(device, &allocInfo, set.p());
			if (result != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate descriptor sets!");
			}
			descs.sets.emplace_back(set);
		}

		std::array<VkWriteDescriptorSet, size> descriptorWrites{};
		descriptorWrites[0] = vkh::createDSWrite(descs.sets[0], 0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageInfos.data(), imageInfos.size());
		descriptorWrites[1] = vkh::createDSWrite(descs.sets[1], 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, lightBufferInfo);
		descriptorWrites[2] = vkh::createDSWrite(descs.sets[2], 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, shadowInfos.data(), shadowInfos.size());
		descriptorWrites[3] = vkh::createDSWrite(descs.sets[3], 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, skyboxInfo);
		descriptorWrites[4] = vkh::createDSWrite(descs.sets[4], 4, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, camMatBufferInfo);
		descriptorWrites[5] = vkh::createDSWrite(descs.sets[5], 5, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, compositionPassImageInfo.data(), compositionPassImageInfo.size());
		descriptorWrites[6] = vkh::createDSWrite(descs.sets[6], 6, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, opaquePassDepthInfo);

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}

	void setupDescriptorSets(bool initial = true) {
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

	void getGLTFImageData(std::shared_ptr<tinygltf::Image>& gltfImage, dvl::Texture& t, unsigned char*& imgData) {
		int width = gltfImage->width;
		int height = gltfImage->height;
		int channels = gltfImage->component;

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
					imgData[(y * width + x) * 4 + c] = gltfImage->image[(y * width + x) * channels + c];
				}
				// if the original image doesn't have an alpha channel, set alpha to 255 (completely opaque)
				if (channels < 4) {
					imgData[(y * width + x) * 4 + 3] = 255;
				}
			}
		}
		imgData = resizeImage(imgData, width, height, t.width, t.height, channels);
		gltfImage.reset();
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

	void createImageStagingBuffer(dvl::Texture& tex, bool cubeMap) {
		auto bpp = cubeMap ? sizeof(float) * 4 : 4;
		VkDeviceSize imageSize = static_cast<VkDeviceSize>(tex.width) * tex.height * bpp;

		if (cubeMap) {
			vkh::createStagingBuffer(tex.stagingBuffer, tex.stagingBufferMem, skyboxData, imageSize, 0);
		}
		else {
			vkh::createStagingBuffer(tex.stagingBuffer, tex.stagingBufferMem, imageData, imageSize, 0);
		}
	}

	void createTexturedImage(dvl::Texture& tex, bool doMipmap, vkh::TextureType type = vkh::BASE) {
		if (!tex.stagingBuffer.valid()) {
			if (tex.path != "gltf") { // standard image
				getImageData(tex.path, imageData);
			}
			else {
				getGLTFImageData(tex.gltfImage, tex, imageData);
			}
			createImageStagingBuffer(tex, false);
			tex.mipLevels = doMipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(tex.width, tex.height)))) + 1 : 1;
			VkFormat imgFormat;
			switch (type) {
			case vkh::BASE:
				imgFormat = VK_FORMAT_R8G8B8A8_SRGB;
				break;
			case vkh::EMISSIVE:
				imgFormat = VK_FORMAT_R8G8B8A8_SRGB;
				break;
			default:
				imgFormat = VK_FORMAT_R8G8B8A8_UNORM;
				break;
			}

			vkh::createImage(tex.image, tex.memory, tex.width, tex.height, imgFormat, tex.mipLevels, 1, false, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, tex.sampleCount);

			// init the VkBufferImageCopy struct for the styaging buffer to image copying
			VkBufferImageCopy region{};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; //specifies the aspect of the image to copy
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount = 1;
			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = { static_cast<uint32_t>(tex.width), static_cast<uint32_t>(tex.height), 1 };

			VkhCommandBuffer tempBuffer = vkh::beginSingleTimeCommands(commandPool);

			vkh::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, tex.mipLevels, 0);
			vkCmdCopyBufferToImage(tempBuffer.v(), tex.stagingBuffer.v(), tex.image.v(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region); //copy the data from the staging buffer to the image

			int mipWidth = tex.width;
			int mipHeight = tex.height;

			// create mipmaps for the image if enabled
			if (doMipmap) {
				for (uint32_t j = 0; j < tex.mipLevels; j++) {
					vkh::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 1, j);

					// if the cutrrent mip level isnt the last, blit the image to generate the next mip level
					// bliting is the process of transfering the image data from one image to another usually with a form of scaling or filtering
					if (j < tex.mipLevels - 1) {
						VkImageBlit blit{};
						blit.srcOffsets[0] = { 0, 0, 0 };
						blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
						blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
						blit.srcSubresource.mipLevel = j;
						blit.srcSubresource.baseArrayLayer = 0;
						blit.srcSubresource.layerCount = 1;
						blit.dstOffsets[0] = { 0, 0, 0 };

						// divide the width and height by 2 if over 1
						blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };

						blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
						blit.dstSubresource.mipLevel = j + 1;
						blit.dstSubresource.baseArrayLayer = 0;
						blit.dstSubresource.layerCount = 1;
						vkCmdBlitImage(tempBuffer.v(), tex.image.v(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex.image.v(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
					}

					vkh::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, j);

					//for the next mip level, divide the width and height by 2, unless they are already 1
					if (mipWidth > 1) mipWidth /= 2;
					if (mipHeight > 1) mipHeight /= 2;
				}
			}
			else {
				vkh::transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, tex.mipLevels, 0);
			}

			vkh::endSingleTimeCommands(tempBuffer, commandPool, graphicsQueue);
			stbi_image_free(imageData);
			imageData = nullptr;
		}
	}

	void getImageDataHDR(std::string path, dvl::Texture& t, float*& imgData) {
		int texWidth, texHeight, texChannels;
		imgData = stbi_loadf(path.c_str(), &texWidth, &texHeight, &texChannels, 4);
		t.width = texWidth;
		t.height = texHeight;
		if (!imgData) {
			std::string error = stbi_failure_reason();
			throw std::runtime_error("failed to load HDR image: " + path + "! Reason: " + error);
		}
	}

	void createTexturedCubemap(dvl::Texture& tex) {
		getImageDataHDR(tex.path, tex, skyboxData);
		if (skyboxData == nullptr) {
			throw std::runtime_error("failed to load image data!");
		}
		createImageStagingBuffer(tex, true);

		// calculate the size of one face of the cubemap
		uint32_t faceWidth = tex.width / 4;
		uint32_t faceHeight = tex.height / 3;
		auto bpp = sizeof(float) * 4; // four floats for R32G32B32A32_SFLOAT
		VkDeviceSize faceSize = static_cast<VkDeviceSize>(faceWidth) * faceHeight * bpp;

		// ensure the atlas dimensions are valid for a horizontal cross layout
		if (faceHeight != faceWidth) {
			throw std::runtime_error("Cubemap atlas dimensions are invalid!!");
		}

		vkh::createImage(tex.image, tex.memory, faceWidth, faceHeight, VK_FORMAT_R32G32B32A32_SFLOAT, 1, 6, true, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, tex.sampleCount);

		vkh::transitionImageLayout(commandPool, tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, 1, 0);
		VkhCommandBuffer copyCmdBuffer = vkh::beginSingleTimeCommands(commandPool);

		std::array<VkBufferImageCopy, 6> regions;
		std::array<std::pair<uint32_t, uint32_t>, 6> faceOffsets = { {{2, 1}, {0, 1}, {1, 0}, {1, 2}, {1, 1}, {3, 1}} };

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

			vkCmdCopyBufferToImage(copyCmdBuffer.v(), tex.stagingBuffer.v(), tex.image.v(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		}
		vkh::endSingleTimeCommands(copyCmdBuffer, commandPool, graphicsQueue);

		vkh::transitionImageLayout(commandPool, tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6, 1, 0);
		stbi_image_free(skyboxData);
		skyboxData = nullptr;
	}

	void createGraphicsPipeline() {
		std::vector<char> vertShaderCode = readFile(SHADER_DIR + "vertex_shader.spv"); //read the vertex shader binary
		std::vector<char> fragShaderCode = readFile(SHADER_DIR + "fragment_shader.spv");
		VkhShaderModule vertShaderModule = vkh::createShaderModule(vertShaderCode);
		VkhShaderModule fragShaderModule = vkh::createShaderModule(fragShaderCode);

		// shader stage setup 
		VkPipelineShaderStageCreateInfo vertShader{}; //vertex shader stage info
		vertShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShader.module = vertShaderModule.v();
		vertShader.pName = "main";

		VkPipelineShaderStageCreateInfo fragShader{}; //fragment shader stage info
		fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShader.module = fragShaderModule.v();
		fragShader.pName = "main";
		VkPipelineShaderStageCreateInfo stages[] = { vertShader, fragShader }; //create an array of the shader stage structs

		// input binding data: tells Vulkan how to read/organize data based on the binding, stride and rate
		VkVertexInputBindingDescription vertBindDesc{};
		vertBindDesc.binding = 0;
		vertBindDesc.stride = sizeof(dvl::Vertex); // number of bytes from one vertex (entry) to the next
		vertBindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // data is per vertex

		VkVertexInputBindingDescription instanceBindDesc{};
		instanceBindDesc.binding = 1;
		instanceBindDesc.stride = sizeof(ModelMat);// number of bytes from one model matrix (entry) to the next
		instanceBindDesc.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE; // data is per instance

		std::array<VkVertexInputBindingDescription, 2> bindDesc = { vertBindDesc, instanceBindDesc };

		// vertex input attributes: defines how the vertex & instance data is structured
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

		// render flag
		attrDesc[9].binding = 1;
		attrDesc[9].location = 9;
		attrDesc[9].format = VK_FORMAT_R32_UINT; // 1 uint32_t
		attrDesc[9].offset = offsetof(ModelMat, render);

		// vertex input state: defines the structure of vertex data for the pipeline
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		// specify the vertex binding descriptions
		// tells the gpu how to move through the vertex buffer
		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindDesc.size());
		vertexInputInfo.pVertexBindingDescriptions = bindDesc.data();

		// specify the vertex attribute descriptions
		// tells the gpu how to interpret individual attributes for each vertex or instance
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
		vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data();

		// input assembly setup: assembles the vertices into primitives
		// in this case, the primitives represent triangles
		VkPipelineInputAssemblyStateCreateInfo inputAssem{};
		inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; //set the topology to triangle list (3 vertices per triangle)
		inputAssem.primitiveRestartEnable = VK_FALSE;

		//scissors setup: defines a region of the framebuffer in which rendering is allowed to happen
		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swap.extent;

		// viewport state: defines how the rendered output is mapped to the framebuffer
		VkPipelineViewportStateCreateInfo vpState{};
		vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		vpState.viewportCount = 1;
		vpState.pViewports = &swapVP;
		vpState.scissorCount = 1;
		vpState.pScissors = &scissor;

		//rasterizer setup: transforms 3D primitives into into fragments to display on the screen
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE; //if true, fragments that are beyond the near and far planes are clamped
		rasterizer.rasterizerDiscardEnable = VK_FALSE; //if true, geometry never passes through the rasterizer and all primitives would be discarded
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //fill the area of the poly with fragments
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; //cull the back faces of triangle
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; //polygons with vertices in counter clockwise order, will be considered front facing
		rasterizer.depthBiasEnable = VK_TRUE; //allows the application of depth bias to fragments

		// multisampling setup: samples multiple points in each pixel and combines them to reduce jagged and blunt edges
		VkPipelineMultisampleStateCreateInfo multiSamp{};
		multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; //number of samples to use per fragment (1 = no mulisampling)
		multiSamp.alphaToCoverageEnable = VK_TRUE; //enables alpha-to-coverage, which uses the alpha component to determine the sample coverage
		multiSamp.alphaToOneEnable = VK_FALSE; //if enabled, forces the alpha component of the color attachment to 1
		multiSamp.sampleShadingEnable = VK_FALSE;// if enabled, would force per sample shading instad of per fragment shading
		multiSamp.minSampleShading = 1.0f; //min fraction for sample shading; closer to one is smoother

		//depth and stencil testing setup: allows for fragments to be discarded based on depth and stencil values
		VkPipelineDepthStencilStateCreateInfo dStencil{};
		dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		dStencil.depthTestEnable = VK_TRUE; //enable depth testing
		dStencil.depthWriteEnable = VK_TRUE; //enable writing to the depth buffer
		dStencil.depthCompareOp = VK_COMPARE_OP_LESS; //fragments with smaller depth values pass the depth test
		dStencil.depthBoundsTestEnable = VK_FALSE; //if true, discards fragments whose depth values fall outside the min and max bounds
		dStencil.minDepthBounds = 0.0f; //min depth bound
		dStencil.maxDepthBounds = 1.0f; //max depth bound
		dStencil.stencilTestEnable = VK_FALSE; // disable stencil testing

		//color blend attachment: tells the gpu how the outputted color from the frag shader will be combined with the data in the framebuffer
		VkPipelineColorBlendAttachmentState colorBA{};
		colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; //color channels to apply the blending operation to
		colorBA.blendEnable = VK_TRUE; //enable blending

		//blending factors for color channels (src = color being output, dst= color already in the framebuffer)
		colorBA.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBA.colorBlendOp = VK_BLEND_OP_ADD;

		//blending factors for the alpha channel (src = alpha being output, dst= alpha already in the framebuffer)
		colorBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBA.alphaBlendOp = VK_BLEND_OP_ADD;

		// color blend state: global pipeline blend settings
		VkPipelineColorBlendStateCreateInfo colorBS{};
		colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBS.logicOpEnable = VK_FALSE; //disable logic operations for color blending
		colorBS.logicOp = VK_LOGIC_OP_COPY;
		colorBS.attachmentCount = 1; //number of color blend attachments
		colorBS.pAttachments = &colorBA;

		// push constants: small pieces of data that can be sent directly to the gpu
		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int) * 2; // 1 int for a bitfield of which textures exist, and 1 int for num of textures in a model

		// pipeline layout setup: defines the connection between shader stages and resources
		// this data includes: descriptorsets and push constants
		VkDescriptorSetLayout setLayouts[] = { descs.layouts[0].v(), descs.layouts[1].v(), descs.layouts[2].v(), descs.layouts[4].v() };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		pipelineLayoutInf.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInf.pushConstantRangeCount = 1;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, opaquePassPipeline.layout.p());
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!!");
		}

		// color attachment: specifies the properties of the color image used in the render pass
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

		// depth attachment: specifies the properties of the depth image used in the render pass
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

		// subpass: a part of a renderpass that contains a set of attachments and specifies the context in which they are used
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; //type of pipeline to bind to
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		//render pass setup: describes the attachments (color & depth) and subpasses used by the pipeline
		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
		VkRenderPassCreateInfo renderPassInf{};
		renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInf.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInf.pAttachments = attachments.data();
		renderPassInf.subpassCount = 1;
		renderPassInf.pSubpasses = &subpass;
		VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, opaquePassPipeline.renderPass.p());
		if (renderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

		//pipeline setup: the data needed to create the pipeline
		VkGraphicsPipelineCreateInfo pipelineInf{};
		pipelineInf.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInf.stageCount = 2; // two shaders - vertex and fragment
		pipelineInf.pStages = stages;
		pipelineInf.pVertexInputState = &vertexInputInfo;
		pipelineInf.pInputAssemblyState = &inputAssem;
		pipelineInf.pViewportState = &vpState;
		pipelineInf.pRasterizationState = &rasterizer;
		pipelineInf.pMultisampleState = &multiSamp;
		pipelineInf.pDepthStencilState = &dStencil;
		pipelineInf.pColorBlendState = &colorBS;
		pipelineInf.layout = opaquePassPipeline.layout.v();
		pipelineInf.renderPass = opaquePassPipeline.renderPass.v();
		pipelineInf.subpass = 0;
		pipelineInf.basePipelineHandle = VK_NULL_HANDLE; // no base pipeline for now
		pipelineInf.basePipelineIndex = -1;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, opaquePassPipeline.pipeline.p());
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}
	}

	void createShadowPipeline() {
		// get shader data
		auto vertShaderSPV = readFile(SHADER_DIR + "shadow_vert_shader.spv");
		auto fragShaderSPV = readFile(SHADER_DIR + "shadow_frag_shader.spv");
		VkhShaderModule shadowVertShaderModule = vkh::createShaderModule(vertShaderSPV);
		VkhShaderModule shadowFragShaderModule = vkh::createShaderModule(fragShaderSPV);

		VkPipelineShaderStageCreateInfo vertStage{};
		vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertStage.module = shadowVertShaderModule.v();
		vertStage.pName = "main";
		VkPipelineShaderStageCreateInfo fragStage{};
		fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragStage.module = shadowFragShaderModule.v();
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
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
		vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssem{};
		inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssem.primitiveRestartEnable = VK_FALSE;

		VkViewport shadowVP{}; // shadow viewport
		shadowVP.x = 0.0f;
		shadowVP.y = 0.0f;
		shadowVP.width = static_cast<float>(shadowProps.width);
		shadowVP.height = static_cast<float>(shadowProps.height);
		shadowVP.minDepth = 0.0f;
		shadowVP.maxDepth = 1.0f;

		VkRect2D shadowScissor{}; // shadow scissor
		shadowScissor.offset = { 0, 0 };
		shadowScissor.extent.width = shadowProps.width;
		shadowScissor.extent.height = shadowProps.height;

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
		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, shadowMapPipeline.renderPass.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow map render pass!");
		}

		VkPipelineColorBlendStateCreateInfo colorBS{};
		colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBS.attachmentCount = 0;

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int); // 1 int for the light index

		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = 1;
		pipelineLayoutInf.pSetLayouts = descs.layouts[1].p();
		pipelineLayoutInf.pushConstantRangeCount = 1; // one range of push constants
		pipelineLayoutInf.pPushConstantRanges = &pushConstantRange;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, shadowMapPipeline.layout.p());
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!!");
		}

		// create the pipeline based off this pipeline and some data from the opaque pipeline
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
		pipelineInfo.layout = shadowMapPipeline.layout.v();
		pipelineInfo.renderPass = shadowMapPipeline.renderPass.v();
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, shadowMapPipeline.pipeline.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow map pipeline!");
		}
	}

	void createSkyboxPipeline() { // same as the normal pipeline, but with a few small changes
		std::vector<char> vertShaderCode = readFile(SHADER_DIR + "sky_vert_shader.spv");
		std::vector<char> fragShaderCode = readFile(SHADER_DIR + "sky_frag_shader.spv");
		VkhShaderModule vertShaderModule = vkh::createShaderModule(vertShaderCode);
		VkhShaderModule fragShaderModule = vkh::createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShader{};
		vertShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShader.module = vertShaderModule.v();
		vertShader.pName = "main";
		VkPipelineShaderStageCreateInfo fragShader{};
		fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShader.module = fragShaderModule.v();
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

		VkDescriptorSetLayout setLayouts[] = { descs.layouts[3].v(), descs.layouts[4].v() };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, skybox.pipelineLayout.p());
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
		pipelineInf.layout = skybox.pipelineLayout.v();
		pipelineInf.renderPass = opaquePassPipeline.renderPass.v();
		pipelineInf.subpass = 0;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, skybox.pipeline.p());
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline for skybox!");
		}
	}

	void createWBOITPipeline() {
		std::vector<char> vertShaderCode = readFile(SHADER_DIR + "wboit_vert_shader.spv");
		std::vector<char> fragShaderCode = readFile(SHADER_DIR + "wboit_frag_shader.spv");
		VkhShaderModule vertShaderModule = vkh::createShaderModule(vertShaderCode);
		VkhShaderModule fragShaderModule = vkh::createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShader{};
		vertShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShader.module = vertShaderModule.v();
		vertShader.pName = "main";
		VkPipelineShaderStageCreateInfo fragShader{};
		fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShader.module = fragShaderModule.v();
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
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
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
		VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, wboitPipeline.renderPass.p());
		if (renderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int) * 2;

		VkDescriptorSetLayout setLayouts[] = { descs.layouts[0].v(), descs.layouts[1].v(), descs.layouts[2].v(), descs.layouts[4].v(), descs.layouts[6].v() };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		pipelineLayoutInf.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInf.pushConstantRangeCount = 1;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, wboitPipeline.layout.p());
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
		pipelineInf.layout = wboitPipeline.layout.v();
		pipelineInf.renderPass = wboitPipeline.renderPass.v();
		pipelineInf.subpass = 0;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, wboitPipeline.pipeline.p());
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline for WBOIT!");
		}
	}

	void createCompositionPipeline() {
		std::vector<char> vertShaderCode = readFile(SHADER_DIR + "composition_vert_shader.spv");
		std::vector<char> fragShaderCode = readFile(SHADER_DIR + "composition_frag_shader.spv");
		VkhShaderModule vertShaderModule = vkh::createShaderModule(vertShaderCode);
		VkhShaderModule fragShaderModule = vkh::createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShader{};
		vertShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShader.module = vertShaderModule.v();
		vertShader.pName = "main";
		VkPipelineShaderStageCreateInfo fragShader{};
		fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShader.module = fragShaderModule.v();
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
		multiSamp.rasterizationSamples = compTex.sampleCount;
		multiSamp.alphaToCoverageEnable = VK_FALSE;
		multiSamp.alphaToOneEnable = VK_FALSE;
		multiSamp.sampleShadingEnable = VK_TRUE;
		multiSamp.minSampleShading = 0.2f;

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
		colorAttachment.samples = compTex.sampleCount;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription colorResolve = {};
		colorResolve.format = swap.imageFormat;
		colorResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorResolveAttachmentRef{};
		colorResolveAttachmentRef.attachment = 1;
		colorResolveAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pResolveAttachments = &colorResolveAttachmentRef;

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, colorResolve };
		VkRenderPassCreateInfo renderPassInf{};
		renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInf.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInf.pAttachments = attachments.data();
		renderPassInf.subpassCount = 1;
		renderPassInf.pSubpasses = &subpass;
		VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, compPipelineData.renderPass.p());
		if (renderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = 1;
		pipelineLayoutInf.pSetLayouts = descs.layouts[5].p();
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, compPipelineData.layout.p());
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
		pipelineInf.layout = compPipelineData.layout.v();
		pipelineInf.renderPass = compPipelineData.renderPass.v();
		pipelineInf.subpass = 0;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, compPipelineData.pipeline.p());
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline for the composition pass!");
		}
	}

	void setupPipelines(bool shadow) {
		opaquePassPipeline.reset();
		compPipelineData.reset();
		skybox.resetPipeline();
		wboitPipeline.reset();

		createGraphicsPipeline();
		createCompositionPipeline();
		createSkyboxPipeline();
		if (shadow) {
			shadowMapPipeline.reset();
			createShadowPipeline();
		}
		createWBOITPipeline();
	}

	static void check_vk_result(VkResult err) { // used to debug imgui errors that have to do with vulkan 
		if (err == 0)
			return;
		std::cerr << "VkResult is " << err << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
		assert(err == 0); // if true, continue, if false, throw error
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

		if (vkCreateDescriptorSetLayout(device, &imguiLayoutInfo, nullptr, imguiDescriptorSetLayout.p()) != VK_SUCCESS) {
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

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, imguiDescriptorPool.p()) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create Imgui descriptor pool!");
		}
	}

	void imguiSetup() {
		// descriptor set creation for imgui:
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
		initInfo.DescriptorPool = imguiDescriptorPool.v();
		initInfo.Allocator = VK_NULL_HANDLE;
		initInfo.MinImageCount = swap.imageCount;
		initInfo.ImageCount = swap.imageCount;
		initInfo.CheckVkResultFn = check_vk_result; // function to check vulkan results
		initInfo.MSAASamples = compTex.sampleCount;
		ImGui_ImplVulkan_Init(&initInfo, compPipelineData.renderPass.v());

		// upload fonts, etc:
		VkhCommandPool guiCommandPool = vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		VkhCommandBuffer guiCommandBuffer = vkh::beginSingleTimeCommands(guiCommandPool);
		ImGui_ImplVulkan_CreateFontsTexture(guiCommandBuffer.v());
		vkh::endSingleTimeCommands(guiCommandBuffer, guiCommandPool, graphicsQueue);
		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}

	void imguiCleanup() {
		ImGui_ImplVulkan_Shutdown();
		ImGui::DestroyContext();
	}

	void setupFences() {
		inFlightFences.resize(swap.images.size());
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // signaled state fence (fence is signaled when created)
		for (size_t i = 0; i < inFlightFences.size(); i++) {
			if (vkCreateFence(device, &fenceInfo, nullptr, inFlightFences[i].p()) != VK_SUCCESS) {
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	void allocateCommandBuffers(CommandBufferSet& cmdBuffers, size_t primaryCount, size_t secondaryCount = 0) {
		cmdBuffers.primary.reserveClear(primaryCount);

		for (size_t i = 0; i < primaryCount; i++) {
			cmdBuffers.primary.pools.emplace_back(vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));
			cmdBuffers.primary.buffers.emplace_back(vkh::allocateCommandBuffers(cmdBuffers.primary.pools[i]));
		}

		if (secondaryCount) {
			cmdBuffers.secondary.reserveClear(secondaryCount);

			for (size_t i = 0; i < secondaryCount; i++) {
				cmdBuffers.secondary.pools.emplace_back(vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT));
				cmdBuffers.secondary.buffers.emplace_back(vkh::allocateCommandBuffers(cmdBuffers.secondary.pools[i], VK_COMMAND_BUFFER_LEVEL_SECONDARY));
			}
		}
	}

	void createCommandBuffers() {
		allocateCommandBuffers(shadowMapCommandBuffers, lights.size(), lights.size());
		allocateCommandBuffers(opaquePassCommandBuffers, swap.imageCount, 1);
		allocateCommandBuffers(wboitCommandBuffers, swap.imageCount, 1);
		allocateCommandBuffers(compCommandBuffers, swap.imageCount, 1);
	}

	void createBLAS(const vkh::BufData& bufferData, size_t index) {
		VkhAccelerationStructure blas{};
		uint32_t primitiveCount = bufferData.indexCount / 3;

		// get the device addresses (location of the data on the device) of the vertex and index buffers
		// the stride and offset are used to go to the starting point of the mesh
		// this allows the data within the gpu to be accessed very efficiently
		VkDeviceAddress vertexAddress = vkh::bufferDeviceAddress(vertBuffer) + (bufferData.vertexOffset * sizeof(dvl::Vertex));
		VkDeviceAddress indexAddress = vkh::bufferDeviceAddress(indBuffer) + (bufferData.indexOffset * sizeof(uint32_t));

		// acceleration structure geometry - specifies the device addresses and data inside of the vertex and index buffers
		VkAccelerationStructureGeometryKHR geometry = {};
		geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geometry.flags = 0; // no geometry flags are set
		geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
		geometry.geometry.triangles.vertexData.deviceAddress = vertexAddress;
		geometry.geometry.triangles.vertexStride = sizeof(dvl::Vertex);
		geometry.geometry.triangles.maxVertex = bufferData.vertexCount;
		geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
		geometry.geometry.triangles.indexData.deviceAddress = indexAddress;

		VkBuildAccelerationStructureFlagsKHR accelerationFlags = 0;
		accelerationFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR; // allows the blas to be compacted
		accelerationFlags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; // optimizes the blas for faster path tracing

		// BLAS build info - specifies the acceleration structure type, the flags, and the geometry
		VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
		buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		buildInfo.flags = accelerationFlags;
		buildInfo.geometryCount = 1;
		buildInfo.pGeometries = &geometry;

		// size requirements for the BLAS - the total size of the acceleration structure, taking into account the amount of primitives, etc
		VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
		sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		vkhfp::vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

		// create a buffer for the BLAS - the buffer used in the creation of the blas
		VkhBuffer blasBuffer;
		VkhDeviceMemory blasMem;
		VkBufferUsageFlags blasUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		vkh::createBuffer(blasBuffer, blasMem, sizeInfo.accelerationStructureSize, blasUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

		// create the BLAS
		VkAccelerationStructureCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		createInfo.buffer = blasBuffer.v();
		createInfo.size = sizeInfo.accelerationStructureSize;
		vkhfp::vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, blas.p());

		// scratch buffer - used to create space for intermediate data thats used when building the BLAS
		VkhBuffer blasScratchBuffer;
		VkhDeviceMemory blasScratchMem;
		VkBufferUsageFlags scratchUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		vkh::createBuffer(blasScratchBuffer, blasScratchMem, sizeInfo.buildScratchSize, scratchUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

		// build range info - specifies the primitive count and offsets for the blas
		VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo = {};
		buildRangeInfo.primitiveCount = primitiveCount;
		buildRangeInfo.primitiveOffset = 0;
		buildRangeInfo.transformOffset = 0;
		buildRangeInfo.firstVertex = 0;
		const VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo = &buildRangeInfo;

		// set the dst of the build info to be the blas and add the scratch buffer address
		buildInfo.dstAccelerationStructure = blas.v();
		buildInfo.scratchData.deviceAddress = vkh::bufferDeviceAddress(blasScratchBuffer);

		// build and populate the BLAS with the geometry data
		VkhCommandBuffer commandBufferB = vkh::beginSingleTimeCommands(commandPool);
		vkhfp::vkCmdBuildAccelerationStructuresKHR(commandBufferB.v(), 1, &buildInfo, &pBuildRangeInfo);
		vkh::endSingleTimeCommands(commandBufferB, commandPool, graphicsQueue);

		// create a query pool used to store the size of the compacted BLAS
		VkhQueryPool queryPool;
		VkQueryPoolCreateInfo queryPoolInfo = {};
		queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolInfo.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
		queryPoolInfo.queryCount = 1;
		vkCreateQueryPool(device, &queryPoolInfo, nullptr, queryPool.p());

		// query the size of the BLAS by writing its properties to the query pool
		// the data becomes avaible after submitting the command buffer
		VkhCommandBuffer commandBufferQ = vkh::beginSingleTimeCommands(commandPool);
		vkCmdResetQueryPool(commandBufferQ.v(), queryPool.v(), 0, 1);
		vkhfp::vkCmdWriteAccelerationStructuresPropertiesKHR(commandBufferQ.v(), 1, blas.p(), VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool.v(), 0);
		vkh::endSingleTimeCommands(commandBufferQ, commandPool, graphicsQueue);

		// get the compacted size from the query pool
		VkDeviceSize compactedSize = 0;
		vkGetQueryPoolResults(device, queryPool.v(), 0, 1, sizeof(VkDeviceSize), &compactedSize, sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);

		// create a buffer for the compacted BLAS
		VkBufferUsageFlags compUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		vkh::createBuffer(BLAS[index].compBuffer, BLAS[index].compMem, compactedSize, compUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

		// create the compacted BLAS
		VkAccelerationStructureCreateInfoKHR compactedCreateInfo = {};
		compactedCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		compactedCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		compactedCreateInfo.buffer = BLAS[index].compBuffer.v();
		compactedCreateInfo.size = compactedSize;
		vkhfp::vkCreateAccelerationStructureKHR(device, &compactedCreateInfo, nullptr, BLAS[index].blas.p());

		// the info for the copying of the original blas to the compacted blas
		VkCopyAccelerationStructureInfoKHR copyInfo = {};
		copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
		copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
		copyInfo.src = blas.v();
		copyInfo.dst = BLAS[index].blas.v();

		// copy the original BLAS to the compacted one
		VkhCommandBuffer commandBufferC = vkh::beginSingleTimeCommands(commandPool);
		vkhfp::vkCmdCopyAccelerationStructureKHR(commandBufferC.v(), &copyInfo);
		vkh::endSingleTimeCommands(commandBufferC, commandPool, graphicsQueue);
	}

	void createMeshInstances() {
		for (size_t i = 0; i < objects.size(); i++) {
			VkAccelerationStructureInstanceKHR meshInstance{};
			size_t bufferInd = modelHashToBufferIndex[objects[i]->meshHash];

			// copy the models model matrix into the instance data
			memcpy(&meshInstance.transform, &objects[i]->modelMatrix, sizeof(VkTransformMatrixKHR));

			VkDeviceAddress blasAddress = vkh::asDeviceAddress(BLAS[bufferInd].blas);

			// populate the instance data
			meshInstance.accelerationStructureReference = blasAddress;
			meshInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
			meshInstance.instanceCustomIndex = static_cast<uint32_t>(i);
			meshInstance.instanceShaderBindingTableRecordOffset = 0;
			meshInstance.mask = 0xFF;
			meshInstances.push_back(meshInstance);
		}
	}


	void createTLAS() {
		if (TLAS.valid()) TLAS.reset();
		// create a buffer to hold all of the instances
		VkDeviceSize iSize = meshInstances.size() * sizeof(VkAccelerationStructureInstanceKHR);
		VkBufferUsageFlags iUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		VkMemoryAllocateFlags iMemFlags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
		vkh::createBuffer(tlasInstanceBuffer, tlasInstanceBufferMem, meshInstances.data(), iSize, iUsage, commandPool, graphicsQueue, iMemFlags, false);

		VkhAccelerationStructure tlas;
		VkDeviceAddress instanceAddress = vkh::bufferDeviceAddress(tlasInstanceBuffer);
		uint32_t primitiveCount = static_cast<uint32_t>(meshInstances.size());

		// acceleration structure geometry
		VkAccelerationStructureGeometryKHR geometry = {};
		geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometry.geometry.instances.data.deviceAddress = instanceAddress;
		geometry.geometry.instances.arrayOfPointers = VK_FALSE;

		VkBuildAccelerationStructureFlagsKHR accelerationFlags = 0;
		accelerationFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR; // allows the tlas to be compacted
		accelerationFlags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; // optimizes the tlas for faster path tracing
		accelerationFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR; // allows the tlas to be updated, without having to fully recreate it

		// TLAS build info - specifies the acceleration structure type, the flags, and the geometry
		VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
		buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		buildInfo.flags = accelerationFlags;
		buildInfo.geometryCount = 1;
		buildInfo.pGeometries = &geometry;

		// size requirements for the TLAS - the total size of the acceleration structure, taking into account the amount of primitives, etc
		VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
		sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		vkhfp::vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

		// create a buffer for the TLAS - the buffer used in the creation of the tlas
		VkBufferUsageFlags tlasUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		vkh::createBuffer(tlasBuffer, tlasMem, sizeInfo.accelerationStructureSize, tlasUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

		// create the TLAS
		VkAccelerationStructureCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		createInfo.buffer = tlasBuffer.v();
		createInfo.size = sizeInfo.accelerationStructureSize;
		vkhfp::vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, tlas.p());

		// scratch buffer - used to create space for intermediate data thats used when building the TLAS
		VkBufferUsageFlags scratchUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		vkh::createBuffer(tlasScratchBuffer, tlasScratchMem, sizeInfo.buildScratchSize, scratchUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

		// build range info - specifies the primitive count and offsets for the tlas
		VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo = {};
		buildRangeInfo.primitiveCount = primitiveCount;
		buildRangeInfo.primitiveOffset = 0;
		buildRangeInfo.transformOffset = 0;
		buildRangeInfo.firstVertex = 0;
		const VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo = &buildRangeInfo;

		// set the dst of the build info to be the tlas and add the scratch buffer address
		buildInfo.dstAccelerationStructure = tlas.v();
		buildInfo.scratchData.deviceAddress = vkh::bufferDeviceAddress(tlasScratchBuffer);

		// build and populate the TLAS
		VkhCommandBuffer commandBufferB = vkh::beginSingleTimeCommands(commandPool);
		vkhfp::vkCmdBuildAccelerationStructuresKHR(commandBufferB.v(), 1, &buildInfo, &pBuildRangeInfo);
		vkh::endSingleTimeCommands(commandBufferB, commandPool, graphicsQueue);

		// create a query pool used to store the size of the compacted TLAS
		VkhQueryPool queryPool;
		VkQueryPoolCreateInfo queryPoolInfo = {};
		queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolInfo.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
		queryPoolInfo.queryCount = 1;
		vkCreateQueryPool(device, &queryPoolInfo, nullptr, queryPool.p());

		// query the size of the TLAS by writing its properties to the query pool
		// the data becomes avaible after submitting the command buffer
		VkhCommandBuffer commandBufferQ = vkh::beginSingleTimeCommands(commandPool);
		vkCmdResetQueryPool(commandBufferQ.v(), queryPool.v(), 0, 1);
		vkhfp::vkCmdWriteAccelerationStructuresPropertiesKHR(commandBufferQ.v(), 1, tlas.p(), VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool.v(), 0);
		vkh::endSingleTimeCommands(commandBufferQ, commandPool, graphicsQueue);

		// get the compacted size from the query pool
		VkDeviceSize compactedSize = 0;
		vkGetQueryPoolResults(device, queryPool.v(), 0, 1, sizeof(VkDeviceSize), &compactedSize, sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);

		// create a buffer for the compacted TLAS
		VkBufferUsageFlags compUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		vkh::createBuffer(compTlasBuffer, compTlasMem, compactedSize, compUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

		// create the compacted TLAS
		VkAccelerationStructureCreateInfoKHR compactedCreateInfo = {};
		compactedCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		compactedCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		compactedCreateInfo.buffer = compTlasBuffer.v();
		compactedCreateInfo.size = compactedSize;
		vkhfp::vkCreateAccelerationStructureKHR(device, &compactedCreateInfo, nullptr, TLAS.p());

		// the info for the copying of the original tlas to the compacted tlas
		VkCopyAccelerationStructureInfoKHR copyInfo = {};
		copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
		copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
		copyInfo.src = tlas.v();
		copyInfo.dst = TLAS.v();

		// copy the original TLAS to the compacted one
		VkhCommandBuffer commandBufferC = vkh::beginSingleTimeCommands(commandPool);
		vkhfp::vkCmdCopyAccelerationStructureKHR(commandBufferC.v(), &copyInfo);
		vkh::endSingleTimeCommands(commandBufferC, commandPool, graphicsQueue);
	}

	void updateTLAS(size_t* objectsIndices, size_t objCount) {
		for (size_t i = 0; i < objCount; i++) {
			size_t index = objectsIndices[i];

			VkAccelerationStructureInstanceKHR meshInstance{};
			size_t bufferInd = modelHashToBufferIndex[objects[index]->meshHash];

			// copy the models model matrix into the instance data
			memcpy(&meshInstance.transform, &objects[index]->modelMatrix, sizeof(VkTransformMatrixKHR));

			// device address of the BLAS
			VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
			addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
			addrInfo.accelerationStructure = BLAS[bufferInd].blas.v();
			VkDeviceAddress blasAddress = vkhfp::vkGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);

			// populate the instance data
			meshInstance.accelerationStructureReference = blasAddress;
			meshInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
			meshInstance.instanceCustomIndex = static_cast<uint32_t>(index);
			meshInstance.instanceShaderBindingTableRecordOffset = 0;
			meshInstance.mask = 0xFF;
			meshInstances.push_back(meshInstance);
		}

		createTLAS();
	}

	void createModelBuffers(bool recreate) {
		std::sort(objects.begin(), objects.end(), [](const auto& a, const auto& b) { return a->meshHash < b->meshHash; });

		if (!recreate) bufferData.resize(getUniqueModels());
		uniqueModelIndex.clear();
		modelHashToBufferIndex.clear();

		VkDeviceSize totalVertexBufferSize = 0;
		VkDeviceSize totalIndexBufferSize = 0;

		// get the total size of the vertex and index buffers
		uint32_t ind = 0;
		for (size_t i = 0; i < objects.size(); i++) {
			auto& obj = objects[i];
			if (uniqueModelIndex.find(obj->meshHash) == uniqueModelIndex.end()) {
				if (!recreate) {
					totalVertexBufferSize += sizeof(dvl::Vertex) * obj->vertices.size();
					totalIndexBufferSize += sizeof(uint32_t) * obj->indices.size();
				}
				uniqueModelIndex[obj->meshHash] = i; //store the index of the object
				modelHashToBufferIndex[obj->meshHash] = ind++;
			}
		}

		if (recreate) return;

		VkhBuffer stagingVertBuffer;
		VkhDeviceMemory stagingVertBufferMem;

		VkhBuffer stagingIndexBuffer;
		VkhDeviceMemory stagingIndexBufferMem;

		const VkMemoryPropertyFlags stagingMemFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		// create and map the vertex buffer
		vkh::createBuffer(stagingVertBuffer, stagingVertBufferMem, totalVertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingMemFlags, 0);
		char* vertexData;
		vkMapMemory(device, stagingVertBufferMem.v(), 0, totalVertexBufferSize, 0, reinterpret_cast<void**>(&vertexData));
		VkDeviceSize currentVertexOffset = 0;

		// create and map the index buffer
		vkh::createBuffer(stagingIndexBuffer, stagingIndexBufferMem, totalIndexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingMemFlags, 0);
		char* indexData;
		vkMapMemory(device, stagingIndexBufferMem.v(), 0, totalIndexBufferSize, 0, reinterpret_cast<void**>(&indexData));
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
		vkUnmapMemory(device, stagingVertBufferMem.v());
		vkUnmapMemory(device, stagingIndexBufferMem.v());

		VkBufferUsageFlags vertU = (rtEnabled) ? (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) : (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		VkBufferUsageFlags indexU = (rtEnabled) ? (VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) : (VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

		VkMemoryAllocateFlags vertM = (rtEnabled) ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;
		VkMemoryAllocateFlags indexM = (rtEnabled) ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;

		vkh::createBuffer(vertBuffer, vertBufferMem, totalVertexBufferSize, vertU, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertM);
		vkh::createBuffer(indBuffer, indBufferMem, totalIndexBufferSize, indexU, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexM);

		VkhCommandBuffer commandBuffer = vkh::beginSingleTimeCommands(commandPool);
		VkBufferCopy copyRegion{};

		// copy the vert staging buffer into the dst vert buffer
		copyRegion.size = totalVertexBufferSize;
		vkCmdCopyBuffer(commandBuffer.v(), stagingVertBuffer.v(), vertBuffer.v(), 1, &copyRegion);

		// copy the index staging buffer into the dst index buffer
		copyRegion.size = totalIndexBufferSize;
		vkCmdCopyBuffer(commandBuffer.v(), stagingIndexBuffer.v(), indBuffer.v(), 1, &copyRegion);

		vkh::endSingleTimeCommands(commandBuffer, commandPool, graphicsQueue);
	}

	void setupAccelerationStructures() {
		if (!rtEnabled) return;
		BLAS.resize(getUniqueModels());

		for (size_t i = 0; i < objects.size(); i++) {
			size_t modelInd = uniqueModelIndex[objects[i]->meshHash];
			if (modelInd != i) continue; // skip if not the first instance of the model
			size_t bufferInd = modelHashToBufferIndex[objects[i]->meshHash];

			createBLAS(bufferData[bufferInd], bufferInd);
		}

		createMeshInstances();
		createTLAS();
	}

	void cloneObject(dml::vec3 pos, uint16_t index, dml::vec3 scale, dml::vec4 rotation) {
		const std::unique_ptr<dvl::Mesh>& other = originalObjects[index];
		dvl::Mesh m;

		m.scale = scale;
		m.position = pos;
		m.startObj = false;
		m.rotation = rotation;
		m.meshHash = other->meshHash;
		m.material = other->material;

		dml::mat4 newModel = dml::translate(pos) * dml::rotateQuat(rotation) * dml::scale(scale);
		m.modelMatrix = newModel * other->modelMatrix;
		objects.push_back(std::make_unique<dvl::Mesh>(std::move(m)));
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

	void summonModel() {
		vkWaitForFences(device, 1, inFlightFences[currentFrame].p(), VK_TRUE, UINT64_MAX);
		if (objects.size() + 2 >= MAX_MODELS) return;
		dml::vec3 pos = dml::getCamWorldPos(cam.viewMatrix);

		cloneObject(pos, 6, { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });
		cloneObject(pos, 10, { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });

		createModelBuffers(true);

		if (rtEnabled) {
			std::array<size_t, 2> indices = { objects.size() - 2, objects.size() - 1 };
			updateTLAS(indices.data(), 2);
		}
		recordSecondaryCommandBuffers();
	}

	void updateLightDS() {
		uint32_t lightSize = static_cast<uint32_t>(lights.size());

		recreateLightBuffer();
		VkDescriptorBufferInfo lightBufferInfo{};
		lightBufferInfo.buffer = lightBuffer.v();
		lightBufferInfo.offset = 0;
		lightBufferInfo.range = lightData.memSize;

		std::array<VkWriteDescriptorSet, 2> dw{};
		dw[0] = vkh::createDSWrite(descs.sets[1], 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, lightBufferInfo);
		dw[1] = vkh::createDSWrite(descs.sets[2], 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, shadowInfos.data(), lightSize);

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(dw.size()), dw.data(), 0, nullptr);
	}

	void summonLight() {
		if (lights.size() + 1 > MAX_LIGHTS) return;
		vkWaitForFences(device, 1, inFlightFences[currentFrame].p(), VK_TRUE, UINT64_MAX);

		dml::vec3 pos = dml::getCamWorldPos(cam.viewMatrix);
		dml::vec3 target = pos + dml::quatToDir(cam.quat);
		Light l = createLight(pos, target);

		vkh::createImage(l.shadowMapData.image, l.shadowMapData.memory, shadowProps.width, shadowProps.height, depthFormat, 1, 1, false, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			l.shadowMapData.sampleCount);
		vkh::createImageView(l.shadowMapData, vkh::DEPTH);
		vkh::createSampler(l.shadowMapData.sampler, l.shadowMapData.mipLevels, vkh::DEPTH);

		vkh::createFB(shadowMapPipeline.renderPass, l.frameBuffer, l.shadowMapData.imageView.p(), 1, shadowProps.width, shadowProps.height);

		VkhCommandPool p = vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		VkhCommandBuffer c = vkh::allocateCommandBuffers(p);

		shadowMapCommandBuffers.primary.pools.push_back(p);
		shadowMapCommandBuffers.primary.buffers.push_back(c);

		VkhCommandPool p2 = vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		VkhCommandBuffer c2 = vkh::allocateCommandBuffers(p2, VK_COMMAND_BUFFER_LEVEL_SECONDARY);

		shadowMapCommandBuffers.secondary.pools.push_back(p2);
		shadowMapCommandBuffers.secondary.buffers.push_back(c2);

		lights.push_back(std::make_unique<Light>(l));

		VkDescriptorImageInfo shadowInfo{};
		shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		shadowInfo.imageView = lights.back()->shadowMapData.imageView.v();
		shadowInfo.sampler = lights.back()->shadowMapData.sampler.v();
		shadowInfos.push_back(shadowInfo);

		updateLightDS();
		recordSecondaryCommandBuffers();
	}

	void recreateLightBuffer() {
		createLightBuffer();
	}

	void recordOpaqueSecondaryCommandBuffers(VkhCommandBuffer& secondary, const PipelineData& pipe, const VkCommandBufferBeginInfo& beginInfo, const VkDescriptorSet* descriptorsets, const size_t descriptorCount, const bool startCommand, const bool endCommand) {
		const std::array<VkBuffer, 2> vertexBuffersArray = { vertBuffer.v(), instanceBuffer.v() };
		const std::array<VkDeviceSize, 2> offsets = { 0, 0 };

		if (startCommand) {
			if (vkBeginCommandBuffer(secondary.v(), &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording opaque secondary command buffer!");
			}
		}

		vkCmdBindPipeline(secondary.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipeline.v());
		vkCmdBindDescriptorSets(secondary.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout.v(), 0, static_cast<uint32_t>(descriptorCount), descriptorsets, 0, nullptr);

		// bind the vertex and instance buffers
		vkCmdBindVertexBuffers(secondary.v(), 0, 2, vertexBuffersArray.data(), offsets.data());
		vkCmdBindIndexBuffer(secondary.v(), indBuffer.v(), 0, VK_INDEX_TYPE_UINT32);

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

				vkCmdPushConstants(secondary.v(), pipe.layout.v(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConst), &pushConst);
				size_t bufferInd = modelHashToBufferIndex[objects[j]->meshHash];
				uint32_t instanceCount = getModelNumHash(objects[uniqueModelInd]->meshHash);
				vkCmdDrawIndexed(secondary.v(), bufferData[bufferInd].indexCount, instanceCount,
					bufferData[bufferInd].indexOffset, bufferData[bufferInd].vertexOffset, uniqueModelInd);
				p++;
			}
		}

		if (endCommand) {
			if (vkEndCommandBuffer(secondary.v()) != VK_SUCCESS) {
				throw std::runtime_error("failed to record opaque secondary command buffer!");
			}
		}
	}

	void recordShadowSecondaryCommandBuffers(std::vector<VkhCommandBuffer>& secondaries, const PipelineData& pipe, const VkCommandBufferBeginInfo& beginInfo, const VkDescriptorSet* descriptorsets, const size_t descriptorCount, const bool startCommand, const bool endCommand) {
		size_t size = secondaries.size();

		const std::array<VkBuffer, 2> vertexBuffersArray = { vertBuffer.v(), instanceBuffer.v() };
		const std::array<VkDeviceSize, 2> offsets = { 0, 0 };

		for (size_t i = 0; i < size; i++) {
			if (startCommand) {
				if (vkBeginCommandBuffer(secondaries[i].v(), &beginInfo) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording shadow secondary command buffer!");
				}
			}

			vkCmdBindPipeline(secondaries[i].v(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipeline.v());
			vkCmdBindDescriptorSets(secondaries[i].v(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout.v(), 0, static_cast<uint32_t>(descriptorCount), descriptorsets, 0, nullptr);

			int lightIndex = static_cast<int>(i);
			vkCmdPushConstants(secondaries[i].v(), shadowMapPipeline.layout.v(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(lightIndex), &lightIndex);

			vkCmdBindVertexBuffers(secondaries[i].v(), 0, 2, vertexBuffersArray.data(), offsets.data());
			vkCmdBindIndexBuffer(secondaries[i].v(), indBuffer.v(), 0, VK_INDEX_TYPE_UINT32);
		}

		// iterate through all objects that cast shadows
		uint32_t p = 0;
		for (size_t j = 0; j < objects.size(); j++) {
			uint32_t uniqueModelInd = static_cast<uint32_t>(uniqueModelIndex[objects[j]->meshHash]);
			if (uniqueModelInd == j) {
				size_t bufferInd = modelHashToBufferIndex[objects[j]->meshHash];
				uint32_t instanceCount = getModelNumHash(objects[uniqueModelInd]->meshHash);
				for (size_t i = 0; i < size; i++) {
					vkCmdDrawIndexed(secondaries[i].v(), bufferData[bufferInd].indexCount, instanceCount,
						bufferData[bufferInd].indexOffset, bufferData[bufferInd].vertexOffset, uniqueModelInd);
				}
				p++;
			}
		}

		if (endCommand) {
			for (size_t i = 0; i < size; i++) {
				if (vkEndCommandBuffer(secondaries[i].v()) != VK_SUCCESS) {
					throw std::runtime_error("failed to record shadow secondary command buffer!");
				}
			}
		}
	}

	void recordCompSecondaryCommandBuffers(VkhCommandBuffer& secondary, const VkCommandBufferBeginInfo& beginInfo, const VkDescriptorSet* descriptorsets, const size_t descriptorCount, const bool startCommand, const bool endCommand) {
		if (startCommand) {
			if (vkBeginCommandBuffer(secondary.v(), &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording composition secondary command buffer!");
			}
		}

		vkCmdBindPipeline(secondary.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, compPipelineData.pipeline.v());
		vkCmdBindDescriptorSets(secondary.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, compPipelineData.layout.v(), 0, static_cast<uint32_t>(descriptorCount), descriptorsets, 0, nullptr);

		vkCmdDraw(secondary.v(), 6, 1, 0, 0);

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
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), secondary.v());

		if (endCommand) {
			if (vkEndCommandBuffer(secondary.v()) != VK_SUCCESS) {
				throw std::runtime_error("failed to record composition secondary command buffer!");
			}
		}
	}

	void recordSecondaryCommandBuffers() {
		const std::array<VkDescriptorSet, 4> opaqueDS = { descs.sets[0].v(), descs.sets[1].v(), descs.sets[2].v(), descs.sets[4].v() };
		std::array<VkDescriptorSet, 5> wboitDS = { descs.sets[0].v(), descs.sets[1].v(), descs.sets[2].v(), descs.sets[4].v(), descs.sets[6].v() };
		const std::array<VkDescriptorSet, 2> skyboxDS = { descs.sets[3].v(), descs.sets[4].v() };

		// FOR THE SHADOW PASS
		VkCommandBufferInheritanceInfo shadowInheritInfo{};
		shadowInheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		shadowInheritInfo.renderPass = shadowMapPipeline.renderPass.v();
		shadowInheritInfo.framebuffer = VK_NULL_HANDLE;
		shadowInheritInfo.subpass = 0;

		VkCommandBufferBeginInfo shadowBeginInfo{};
		shadowBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		shadowBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		shadowBeginInfo.pInheritanceInfo = &shadowInheritInfo;

		recordShadowSecondaryCommandBuffers(shadowMapCommandBuffers.secondary.buffers, shadowMapPipeline, shadowBeginInfo, descs.sets[1].p(), 1, true, true);

		// FOR THE OPAQUE & SKYBOX PASS
		VkCommandBufferInheritanceInfo opaqueInheritInfo{};
		opaqueInheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		opaqueInheritInfo.renderPass = opaquePassPipeline.renderPass.v();
		opaqueInheritInfo.framebuffer = VK_NULL_HANDLE;
		opaqueInheritInfo.subpass = 0;

		VkCommandBufferBeginInfo opaqueBeginInfo{};
		opaqueBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		opaqueBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		opaqueBeginInfo.pInheritanceInfo = &opaqueInheritInfo;

		VkhCommandBuffer& secondary = opaquePassCommandBuffers.secondary[0];
		if (vkBeginCommandBuffer(secondary.v(), &opaqueBeginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording secondary command buffer!");
		}

		const VkDeviceSize skyboxOffset = 0;
		vkCmdBindPipeline(secondary.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipeline.v());
		vkCmdBindDescriptorSets(secondary.v(), VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipelineLayout.v(), 0, static_cast<uint32_t>(skyboxDS.size()), skyboxDS.data(), 0, nullptr);
		vkCmdBindVertexBuffers(secondary.v(), 0, 1, skybox.vertBuffer.p(), &skyboxOffset);
		vkCmdBindIndexBuffer(secondary.v(), skybox.indBuffer.v(), 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(secondary.v(), skybox.bufferData.indexCount, 1, skybox.bufferData.indexOffset, skybox.bufferData.vertexOffset, 0);

		recordOpaqueSecondaryCommandBuffers(secondary, opaquePassPipeline, opaqueBeginInfo, opaqueDS.data(), opaqueDS.size(), false, true);

		// FOR THE WBOIT PASS
		VkCommandBufferInheritanceInfo wboitInheritInfo{};
		wboitInheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		wboitInheritInfo.renderPass = wboitPipeline.renderPass.v();
		wboitInheritInfo.framebuffer = VK_NULL_HANDLE;
		wboitInheritInfo.subpass = 0;

		VkCommandBufferBeginInfo wboitBeginInfo{};
		wboitBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		wboitBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		wboitBeginInfo.pInheritanceInfo = &wboitInheritInfo;

		recordOpaqueSecondaryCommandBuffers(wboitCommandBuffers.secondary[0], wboitPipeline, wboitBeginInfo, wboitDS.data(), wboitDS.size(), true, true);
	}


	void recordShadowCommandBuffers() {
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
				VkhCommandBuffer& shadowCommandBuffer = shadowMapCommandBuffers.primary.buffers[i];
				VkhCommandBuffer& secondary = shadowMapCommandBuffers.secondary.buffers[i];

				if (vkBeginCommandBuffer(shadowCommandBuffer.v(), &beginInfoP) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				// render pass
				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = shadowMapPipeline.renderPass.v();
				renderPassInfo.framebuffer = lights[i]->frameBuffer.v();
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = { shadowProps.width, shadowProps.height };
				renderPassInfo.clearValueCount = 1;
				renderPassInfo.pClearValues = &clearValue;
				vkCmdBeginRenderPass(shadowCommandBuffer.v(), &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

				vkCmdExecuteCommands(shadowCommandBuffer.v(), 1, secondary.p());

				// end the render pass and command buffer
				vkCmdEndRenderPass(shadowCommandBuffer.v());
				if (vkEndCommandBuffer(shadowCommandBuffer.v()) != VK_SUCCESS) {
					throw std::runtime_error("failed to record command buffer!");
				}
#ifdef PROFILE_COMMAND_BUFFERS
#else
				});
#endif
		}
	}

	void recordOpaqueCommandBuffers() {
		const std::array<VkClearValue, 2> clearValues = { VkClearValue{0.18f, 0.3f, 0.30f, 1.0f}, VkClearValue{1.0f, 0} };

		VkCommandBufferBeginInfo beginInfoP{};
		beginInfoP.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfoP.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfoP.pInheritanceInfo = nullptr;
		for (size_t i = 0; i < swap.images.size(); i++) {
#ifdef PROFILE_COMMAND_BUFFERS
#else
			tfCmd.emplace([&, i, clearValues, beginInfoP]() {
#endif
				VkhCommandBuffer& opaqueCommandBuffer = opaquePassCommandBuffers.primary[i];
				if (vkBeginCommandBuffer(opaqueCommandBuffer.v(), &beginInfoP) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = opaquePassPipeline.renderPass.v();
				renderPassInfo.framebuffer = opaquePassFB.v();
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = swap.extent;
				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				vkCmdBeginRenderPass(opaqueCommandBuffer.v(), &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
				vkCmdExecuteCommands(opaqueCommandBuffer.v(), static_cast<uint32_t>(opaquePassCommandBuffers.secondary.size()), opaquePassCommandBuffers.secondary.data());
				vkCmdEndRenderPass(opaqueCommandBuffer.v());
				if (vkEndCommandBuffer(opaqueCommandBuffer.v()) != VK_SUCCESS) {
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
		VkDeviceSize offset[] = { 0, 0 };

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		for (size_t i = 0; i < swap.images.size(); i++) {
#ifdef PROFILE_COMMAND_BUFFERS
#else
			tfCmd.emplace([&, i, clearValues, offset, beginInfo]() {
#endif
				VkhCommandBuffer& wboitCommandBuffer = wboitCommandBuffers.primary[i];
				if (vkBeginCommandBuffer(wboitCommandBuffer.v(), &beginInfo) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = wboitPipeline.renderPass.v();
				renderPassInfo.framebuffer = wboit.frameBuffer.v();
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = swap.extent;
				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				vkh::transitionImageLayout(wboitCommandBuffer, opaquePassTextures.depth.image, depthFormat, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, 0);

				vkCmdBeginRenderPass(wboitCommandBuffer.v(), &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
				vkCmdExecuteCommands(wboitCommandBuffer.v(), static_cast<uint32_t>(wboitCommandBuffers.secondary.size()), wboitCommandBuffers.secondary.data());
				vkCmdEndRenderPass(wboitCommandBuffer.v());

				if (vkEndCommandBuffer(wboitCommandBuffer.v()) != VK_SUCCESS) {
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
		VkhDescriptorSet compDS = descs.sets[5];

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		VkCommandBufferInheritanceInfo inheritInfo{};
		inheritInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		inheritInfo.renderPass = compPipelineData.renderPass.v();
		inheritInfo.framebuffer = VK_NULL_HANDLE;
		inheritInfo.subpass = 0;

		VkCommandBufferBeginInfo beginInfoS{};
		beginInfoS.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfoS.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
		beginInfoS.pInheritanceInfo = &inheritInfo;

		recordCompSecondaryCommandBuffers(compCommandBuffers.secondary[0], beginInfoS, compDS.p(), 1, true, true);

		for (size_t i = 0; i < swap.images.size(); i++) {
#ifdef PROFILE_COMMAND_BUFFERS
#else
			tfCmd.emplace([&, i, clearValues, beginInfo]() {
#endif
				VkhCommandBuffer& compCommandBuffer = compCommandBuffers.primary.buffers[i];
				if (vkBeginCommandBuffer(compCommandBuffer.v(), &beginInfo) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = compPipelineData.renderPass.v();
				renderPassInfo.framebuffer = swap.framebuffers[i].v();
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = swap.extent;
				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				vkCmdBeginRenderPass(compCommandBuffer.v(), &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
				vkCmdExecuteCommands(compCommandBuffer.v(), 1, compCommandBuffers.secondary[0].p());
				vkCmdEndRenderPass(compCommandBuffer.v());
				if (vkEndCommandBuffer(compCommandBuffer.v()) != VK_SUCCESS) {
					throw std::runtime_error("failed to record command buffer!");
				}
#ifdef PROFILE_COMMAND_BUFFERS
#else
				});
#endif
		}
	}

	void drawText(std::string text, float x, float y, ImFont* font = nullptr, ImVec4 bgColor = ImVec4(-1, -1, -1, -1)) {
		// set the pos and size of the window
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

		// if font exists, use it
		// otherwise, use the default font
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

		// if a custom background was set, revert it
		if (bgColor.x != -1) {
			ImGui::PopStyleColor();
		}
		ImGui::End();
	}

	void createFrameBuffers(bool initial) {
		if (initial) {
			// create the shadowmap framebuffers
			for (size_t i = 0; i < lights.size(); i++) {
				vkh::createFB(shadowMapPipeline.renderPass, lights[i]->frameBuffer, lights[i]->shadowMapData.imageView.p(), 1, shadowProps.width, shadowProps.height);
			}
		}

		// create the opaque pass framebuffer
		std::vector<VkImageView> attachmentsM = { opaquePassTextures.color.imageView.v(), opaquePassTextures.depth.imageView.v() };
		vkh::createFB(opaquePassPipeline.renderPass, opaquePassFB, attachmentsM.data(), attachmentsM.size(), swap.extent.width, swap.extent.height);

		// create the wboit framebuffer
		vkh::createFB(wboitPipeline.renderPass, wboit.frameBuffer, wboit.weightedColor.imageView.p(), 1, swap.extent.width, swap.extent.height);

		// create the composition framebuffers
		size_t swapSize = swap.imageViews.size();
		if (initial) swap.framebuffers.resize(swapSize);
		for (size_t i = 0; i < swapSize; i++) {
			std::vector<VkImageView> attachments = { compTex.imageView.v(), swap.imageViews[i].v() };
			vkh::createFB(compPipelineData.renderPass, swap.framebuffers[i], attachments.data(), attachments.size(), swap.extent.width, swap.extent.height);
		}
	}

	void createSemaphores() {
		vkh::createSemaphore(imageAvailableSemaphore);
		vkh::createSemaphore(renderFinishedSemaphore);
		vkh::createSemaphore(shadowSemaphore);
		vkh::createSemaphore(wboitSemaphore);
		vkh::createSemaphore(compSemaphore);
	}

	void recreateSwap() {
		std::cout << "Recreating swap chain..." << std::endl;
		int width = 0, height = 0;
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}
		vkWaitForFences(device, 1, inFlightFences[currentFrame].p(), VK_TRUE, UINT64_MAX);
		vkDeviceWaitIdle(device); // wait for thr device to be idle

		swap.swapChain.reset(); // reset the SC

		createSC();
		createSCImageViews();
		setupTextures();

		// create the descriptorsets
		setupDescriptorSets(false);

		// create the pipelines
		setupPipelines(false);

		// create the framebuffers
		createFrameBuffers(false);

		initializeMouseInput(true);

		recordSecondaryCommandBuffers();
	}

	void drawFrame() {
		uint32_t imageIndex;
		vkResetFences(device, 1, inFlightFences[currentFrame].p());

		// acquire the next image from the swapchain
		VkResult result = vkAcquireNextImageKHR(device, swap.swapChain.v(), std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore.v(), VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			vkDeviceWaitIdle(device);
			recreateSwap();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		std::vector<VkSubmitInfo> submitInfos;

		for (VkhCommandBuffer& shadow : shadowMapCommandBuffers.primary.buffers) {
			VkSubmitInfo sub = vkh::createSubmitInfo(&shadow, 1);
			submitInfos.push_back(sub);
		}

		submitInfos.push_back(vkh::createSubmitInfo(&opaquePassCommandBuffers.primary[imageIndex], 1, waitStages, imageAvailableSemaphore, wboitSemaphore));
		submitInfos.push_back(vkh::createSubmitInfo(&wboitCommandBuffers.primary[imageIndex], 1, waitStages, wboitSemaphore, compSemaphore));
		submitInfos.push_back(vkh::createSubmitInfo(&compCommandBuffers.primary[imageIndex], 1, waitStages, compSemaphore, renderFinishedSemaphore));

		// submit all command buffers in a single call
		if (vkQueueSubmit(graphicsQueue, static_cast<uint32_t>(submitInfos.size()), submitInfos.data(), inFlightFences[currentFrame].v()) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit command buffers!");
		}

		// present the image
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = renderFinishedSemaphore.p();
		VkSwapchainKHR swapChains[] = { swap.swapChain.v() };
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
		vkWaitForFences(device, 1, inFlightFences[currentFrame].p(), VK_TRUE, UINT64_MAX);

#ifdef PROFILE_COMMAND_BUFFERS
		utils::sep();

		auto now = utils::now();
		recordShadowCommandBuffers();
		auto duration = utils::duration<microseconds>(now);
		std::cout << "recordShadowCommandBuffers: " << utils::durationString(duration) << std::endl;

		now = utils::now();
		recordOpaqueCommandBuffers();
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
		recordOpaqueCommandBuffers();
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
			handleKeyboardInput();
			recordAllCommandBuffers();
			updateUBO();
			calcFps(startTime, previousTime, frameCount);
#endif
		}

		vkDeviceWaitIdle(device);
		imguiCleanup();
	}

	void initializeMouseInput(bool initial) {
		// set the lastX and lastY to the center of the screen
		if (initial) {
			mouse.lastX = static_cast<float>(swap.extent.width) / 2.0f;
			mouse.lastY = static_cast<float>(swap.extent.height) / 2.0f;
			glfwSetCursorPos(window, mouse.lastX, mouse.lastY);
		}

		// only hide and capture cursor if cam.locked is true
		if (mouse.locked) {
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

		if (mouse.locked) {
			float xoff = mouse.lastX - xp;
			float yoff = mouse.lastY - yp;
			mouse.lastX = xp;
			mouse.lastY = yp;

			float sens = 0.1f;
			xoff *= sens;
			yoff *= sens;

			mouse.rightAngle -= xoff;
			mouse.upAngle -= yoff;
		}
	}

	void handleKeyboardInput() {
		double currentFrame = glfwGetTime();
		float deltaTime = static_cast<float>(currentFrame - lastFrame);
		lastFrame = currentFrame;

		float cameraSpeed = 2.0f * deltaTime;

		mouse.upAngle = fmod(mouse.upAngle + 360.0f, 360.0f);
		mouse.rightAngle = fmod(mouse.rightAngle + 360.0f, 360.0f);

		cam.updateQuaternion(mouse);
		dml::vec3 forward = dml::quatToDir(cam.quat);
		dml::vec3 right = dml::normalize(dml::cross(forward, dml::vec3(0, 1, 0)));

		// camera movement
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			cam.pos -= forward * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			cam.pos += forward * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			cam.pos += right * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			cam.pos -= right * cameraSpeed;
		}

		// up and down movement
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
			cam.pos.y -= 1 * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
			cam.pos.y += 1 * cameraSpeed;
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
			size_t vertCount = 0;
			for (const auto& o : objects) {
				vertCount += o->vertices.size();
			}

			double score = fps * (((vertCount) / 50000.0) + std::pow(lights.size(), 1.3) + (objects.size() / 10.0));

			utils::sep();
			std::cout << "Vertex count: " << vertCount << std::endl;
			std::cout << "Object count: " << objects.size() << std::endl;
			std::cout << "Light count: " << lights.size() << " / " << MAX_LIGHTS << std::endl;
			std::cout << "Score: " << score << std::endl;
		}

		// lock / unlock mouse
		if (escapeKey.isPressed()) {
			mouse.locked = !mouse.locked;
			initializeMouseInput(mouse.locked);
		}
	}

	void initVulkan() {
		// initialize Vulkan components
		auto now = utils::now();
		validateFiles();
		createInstance();
		createSurface();
		pickDevice();
		createLogicalDevice();
		initQueues();
		createSC();
		setupFences();
		createSemaphores();

		commandPool = vkh::createCommandPool(queueFamilyIndices.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		initializeMouseInput(true);
		loadUniqueObjects();

		// create buffers and textures
		createModelBuffers(false);
		setupAccelerationStructures();
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
		recordSecondaryCommandBuffers();
		recordAllCommandBuffers();

		auto duration = utils::duration<milliseconds>(now);
		utils::printDuration(duration);

		std::cout << "Vulkan initialized successfully! Unique models: " << getUniqueModels() << std::endl;
	}
};

int main() {
	{
		Engine app;
		try {
			app.run();
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}
	}

	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyDevice(device, nullptr);
	vkDestroyInstance(instance, nullptr);

	utils::sep();
	utils::sep();
	std::cout << "CLOSING ENGINE" << std::endl;
	utils::sep();
	utils::sep();

	return EXIT_SUCCESS;
}