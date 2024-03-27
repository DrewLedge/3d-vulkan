#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../ext/tiny_gltf.h" // load .obj and .mtl files
#include "../ext/stb_image_resize.h"
#include "dml.hpp"
#include "dvl.hpp"
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
#include <chrono> // random seed based on time
#include <cmath>
#include <memory>

#define MAX_TEXTURES 4000 // temp max num of textures and models (used for passing data to shaders)
#define MAX_MODELS 1200

#define SCREEN_WIDTH 3200
#define SCREEN_HEIGHT 1800

const std::string SHADER_DIR = "shaders/compiled/";
const std::string MODEL_DIR = "assets/models/";
const std::string SKYBOX_DIR = "assets/skyboxes/";
const std::string FONT_DIR = "assets/fonts/";

bool rtSupported = false; // a bool if raytracing is supported on the device


struct camData {
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

	camData() {
		camPos = { 0.0f, 0.0f, 0.0f };
		quat = { 0.0f, 0.0f, 0.0f, 1.0f };
		rightAngle = 0.0f;
		upAngle = 0.0f;
		lastX = 0.0f;
		lastY = 0.0f;
		locked = true;
	}

	dml::mat4 getOrientation() const {
		const float dr = PI / 180.0f;
		dml::vec4 yRot = dml::angleAxis(upAngle * dr, dml::vec3(1, 0, 0));
		dml::vec4 xRot = dml::angleAxis(rightAngle * dr, dml::vec3(0, 1, 0));
		dml::vec4 q = yRot * xRot;

		return dml::rotateQ(q).transpose(); // convert the quaternion to a rotation matrix
	}

	dml::vec3 getLookPoint() const {
		return camPos + getOrientation() * dml::vec3(0, 0, -1);
	}

	dml::mat4 getViewMatrix() {
		const float dr = PI / 180.0f;
		return dml::viewMatrix(camPos, upAngle * dr, rightAngle * dr);
	}
};

camData cam;

class Engine {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}
private:
	struct Texture {
		VkSampler sampler;
		VkImage image;
		VkDeviceMemory memory;
		VkImageView imageView;
		std::string path;
		uint32_t mipLevels;
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMem;
		tinygltf::Image gltfImage;
		bool found;
		uint16_t width;
		uint16_t height;

		Texture()
			: sampler(VK_NULL_HANDLE),
			image(VK_NULL_HANDLE),
			memory(VK_NULL_HANDLE),
			imageView(VK_NULL_HANDLE),
			path(""),
			mipLevels(1),
			stagingBuffer(VK_NULL_HANDLE),
			stagingBufferMem(VK_NULL_HANDLE),
			gltfImage(),
			found(false),
			width(1024),
			height(1024)
		{}

		bool operator==(const Texture& other) const {
			return sampler == other.sampler
				&& image == other.image
				&& memory == other.memory
				&& imageView == other.imageView
				&& path == other.path
				&& mipLevels == other.mipLevels
				&& stagingBuffer == other.stagingBuffer
				&& stagingBufferMem == other.stagingBufferMem
				&& width == other.width
				&& height == other.height;

		}
	};

	struct texHash {
		size_t operator()(const Texture& tex) const {
			size_t seed = 0;
			seed ^= std::hash<VkImage>{}(tex.image) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<VkSampler>{}(tex.sampler) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<VkDeviceMemory>{}(tex.memory) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<VkImageView>{}(tex.imageView) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<std::string>{}(tex.path) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<uint32_t>{}(tex.mipLevels) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<VkBuffer>{}(tex.stagingBuffer) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<VkDeviceMemory>{}(tex.stagingBufferMem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<uint16_t>{}(tex.width) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<uint16_t>{}(tex.height) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			return seed;
		}
	};

	struct Materials {
		Texture metallicRoughness;
		Texture baseColor;
		Texture normalMap;
		Texture occlusionMap;
		Texture emissiveMap;
	};
	struct model {
		Materials material; //used to store all the textures/materials of the model
		std::vector<dvl::Vertex> vertices;
		std::vector<uint32_t> indices;
		std::string pathObj; // i.e "models/cube.obj"

		dml::vec3 position;  // position of the model
		dml::vec4 rotation;  // rotation of the model in quaternions
		dml::vec3 scale;     // scale of the model
		dml::mat4 modelMatrix;

		size_t textureCount; // number of textures in the model
		size_t texIndex; // where in the texture array the textures of the model start

		bool startObj; // wether is loaded at the start of the program or not
		bool player; // if the object is treated as a player model or not

		size_t modelHash;
		std::string name;

		// default constructor
		model()
			: material(),
			vertices(),
			indices(),
			pathObj(""),
			position(dml::vec3(0.0f, 0.0f, 0.0f)),  // set default position to origin
			rotation(dml::vec4(0.0f, 0.0f, 0.0f, 0.0f)),  // set default rotation to no rotation
			scale(dml::vec3(0.1f, 0.1f, 0.1f)),
			modelMatrix(),
			textureCount(0),
			texIndex(0),
			startObj(true),
			player(false),
			modelHash(),
			name("")
		{}

		// copy constructor
		model(const model& other)
			: material(other.material),
			vertices(other.vertices),
			indices(other.indices),
			pathObj(other.pathObj),
			position(other.position),
			rotation(other.rotation),
			scale(other.scale),
			modelMatrix(other.modelMatrix),
			textureCount(other.textureCount),
			texIndex(other.texIndex),
			startObj(other.startObj),
			player(other.player),
			modelHash(other.modelHash),
			name(other.name) {
		}
	};
	struct shadowMapDataObject {
		VkImage image;
		VkImageView imageView;
		VkSampler sampler;
		VkFramebuffer frameBuffer;
		uint32_t mipLevels; // placeholder (not used)
		VkDeviceMemory memory;

		shadowMapDataObject() {
			image = VK_NULL_HANDLE;
			imageView = VK_NULL_HANDLE;
			sampler = VK_NULL_HANDLE;
			frameBuffer = VK_NULL_HANDLE;
			mipLevels = 0;
			memory = VK_NULL_HANDLE;
		}
	};

	struct light { // spotlight
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
		shadowMapDataObject shadowMapData;

		// default constructor
		light()
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
			shadowMapData() {
		}

		// copy constructor
		light(const light& other)
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
			shadowMapData(other.shadowMapData) {
		}
	};
	struct lightMatrixUBO {
		dml::mat4 view;
		dml::mat4 proj;
	};

	struct lightCords {
		dml::vec3 pos;
		dml::vec3 col;
		dml::vec3 target;
		float baseIntensity;
		float innerConeAngle; // in degrees
		float outerConeAngle; // in degrees
		float constantAttenuation;
		float linearAttenuation;
		float quadraticAttenuation;
	};

	struct lightDataSSBO {
		lightMatrixUBO lightsMatricies[20]; // max 20 lights
		lightCords lightCords[20]; // max 20 lights
	};


	struct modelMat {
		dml::mat4 model;
		uint32_t render;
	};
	struct modelMatInstanceData {
		modelMat object[MAX_MODELS];
	};
	struct camUBO {
		dml::mat4 view;
		dml::mat4 proj;
	};

	struct shadowMapProportionsObject {
		uint32_t mapWidth = 2048;
		uint32_t mapHeight = 2048;
	};

	struct bufData {
		uint32_t vertexOffset;
		uint32_t vertexCount;
		uint32_t indexOffset;
		uint32_t indexCount;
	};

	struct sBox { // skybox struct
		Texture cubemap;
		Texture out;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		bufData bufferData; // buffer data for the skybox (vert offsets, etc)
		VkBuffer vertBuffer;
		VkDeviceMemory vertBufferMem;
		VkBuffer indBuffer;
		VkDeviceMemory indBufferMem;
		std::vector<dml::vec3> vertices;
		std::vector<uint32_t> indices;

		sBox() {
			indices = {
				0, 1, 2, 2, 3, 0,
				7, 6, 5, 5, 4, 7,
				4, 5, 1, 1, 0, 4,
				3, 2, 6, 6, 7, 3,
				4, 0, 3, 3, 7, 4,
				1, 5, 6, 6, 2, 1
			};

			vertices = {
				{-1.0f,  1.0f,  1.0f},
				{-1.0f, -1.0f,  1.0f},
				{ 1.0f, -1.0f,  1.0f},
				{ 1.0f,  1.0f,  1.0f},
				{-1.0f,  1.0f, -1.0f},
				{-1.0f, -1.0f, -1.0f},
				{ 1.0f, -1.0f, -1.0f},
				{ 1.0f,  1.0f, -1.0f}
			};
		}
	};
	struct descriptorSetObject {
		std::vector<VkDescriptorSetLayout> layouts;
		std::vector<VkDescriptorSet> sets;
		std::vector<VkDescriptorPool> pools;
	};

	struct pipelineData {
		VkRenderPass renderPass;
		VkPipelineLayout layout;
		VkPipeline graphicsPipeline;
	};

	struct swapChainData {
		VkSwapchainKHR swapChain;
		std::vector<VkImage> images;
		VkFormat imageFormat;
		VkExtent2D extent;
		std::vector<VkImageView> imageViews;
		uint32_t imageCount;
		std::vector<VkFramebuffer> framebuffers;
	};

	struct keyPressObj { // prevent a certain key from being held down
		bool escPressedLastFrame = false; // unlock mouse
	};

	struct WBOITData { // weighted blended order independent transparency
		Texture weightedColor;
		Texture weightedAlpha;

		VkFramebuffer frameBuffer;
		pipelineData pipeline;
	};

	struct mainPassTex {
		Texture depth;
		Texture color;
	};

	// window and rendering context
	GLFWwindow* window;
	VkSurfaceKHR surface;
	VkInstance instance;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	VkQueue presentQueue;
	VkQueue graphicsQueue;
	keyPressObj keyPO;

	// swap chain and framebuffers
	swapChainData swap;
	size_t currentFrame = 0;

	// viewport config
	VkViewport vp{};

	// rendering pipeline data
	pipelineData mainPassPipeline;
	VkFramebuffer mainPassFB;
	mainPassTex mainPassTextures;

	pipelineData shadowMapPipeline;
	pipelineData compositionPipelineData;

	// command buffers and command pool
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> mainPassCommandBuffers;
	std::vector<VkCommandBuffer> shadowMapCommandBuffers;
	VkCommandBuffer wboitCommandBuffer;
	std::vector<VkCommandBuffer> compCommandBuffers;

	// buffers and related memory
	VkBuffer vertBuffer;
	VkDeviceMemory vertBufferMem;

	VkBuffer indBuffer;
	VkDeviceMemory indBufferMem;

	VkBuffer instanceBuffer;
	VkDeviceMemory instanceBufferMem;

	VkBuffer lightBuffer;
	VkDeviceMemory lightBufferMem;

	VkBuffer sceneIndexBuffer;
	VkDeviceMemory sceneIndexBufferMem;

	// synchronization primitives
	std::vector<VkFence> inFlightFences;
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore renderFinishedSemaphore;
	VkSemaphore shadowSemaphore;
	VkSemaphore wboitSemaphore;
	VkSemaphore compSemaphore;

	// shader modules
	VkShaderModule fragShaderModule;
	VkShaderModule vertShaderModule;

	// depth buffering
	VkFormat depthFormat;
	WBOITData wboit;

	// descriptor sets and pools
	descriptorSetObject descs;
	VkDescriptorSetLayout imguiDescriptorSetLayout;
	VkDescriptorPool imguiDescriptorPool;

	// scene data and objects
	std::vector<bufData> bufferData;
	std::vector<std::unique_ptr<model>> objects;
	std::vector<std::unique_ptr<model>> originalObjects;
	std::vector<uint32_t> playerModels;
	modelMatInstanceData objInstanceData;
	camUBO camMatData;
	lightDataSSBO lightData;
	std::vector<std::unique_ptr<light>> lights;
	shadowMapProportionsObject shadowProps;
	uint32_t modelIndex; // index of where vertecies are loaded to

	std::unordered_map<size_t, size_t> uniqueModelIndex;
	std::unordered_map<size_t, size_t> modelHashToBufferIndex;

	// textures and materials
	std::vector<Texture> allTextures;
	std::vector<int> meshTexStartInd;
	size_t totalTextureCount = 0;
	unsigned char* imageData;

	// skybox data
	float* skyboxData;
	sBox skybox;

	// font data
	ImFont* font_small;
	ImFont* font_large;

	// performance metrics
	uint32_t fps;
	double lastFrame = 0.0;

	// mutexes for multithreading
	std::mutex modelMtx;
	std::mutex mtx;

	std::array<float, 3> toArray(const dml::vec3& v) {
		return { v.x, v.y, v.z };
	}

	void sep() {
		std::cout << "---------------------------------" << std::endl;
	}

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
	const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
	};
	int rng(int min, int max) {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(min, max);
		return dist(gen);
	};

	void debugStruct(model stru) {
		sep();
		std::cout << "model: " << stru.pathObj << std::endl;
		std::cout << "vertices: " << stru.vertices.size() << std::endl;
		std::cout << "indices: " << stru.indices.size() << std::endl;
		sep();
	}
	void createObject(std::string path, dml::vec3 scale, dml::vec4 rotation, dml::vec3 pos) {
		loadModel(scale, pos, rotation, MODEL_DIR + path);
	}
	void createLight(dml::vec3 pos, dml::vec3 color, float intensity, dml::vec3 t) {
		light l;
		l.col = color;
		l.pos = pos;
		l.baseIntensity = intensity;
		l.target = t;
		l.constantAttenuation = 1.0f;
		l.linearAttenuation = 0.1f;
		l.quadraticAttenuation = 0.032f;
		l.innerConeAngle = 6.6f;
		l.outerConeAngle = 10.0f;
		lights.push_back(std::make_unique<light>(l));
	}
	void setPlayer(uint16_t i) {
		auto p = std::make_unique<model>(*objects[i]);
		p->player = true;
		p->scale = dml::vec3(0.3f, 0.3f, 0.3f);
		p->position = dml::vec3(-3.0f, 0.0f, 3.0f);
		playerModels.push_back(i);
		objects.push_back(std::move(p));
	}

	void loadUniqueObjects() { // load all unqiue objects and all lights
		//createObject("models/sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
		//createObject("models/sword.glb", { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
		createObject("sword.glb", { 103.2f, 103.2f, 103.2f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f });
		createObject("knight.glb", { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.23f, 0.0f, 2.11f });
		createObject("knight.glb", { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f });
		createObject("sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQ({ 3.0f, 1.0f, -2.11f }, { 0.0f, 0.0f, 0.0f }), { 3.0f, 1.0f, -2.11f });
		createObject("sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQ({ -2.0f, 0.0f, 2.11f }, { 0.0f, 0.0f, 0.0f }), { -2.0f, 0.0f, 2.11f });
		createObject("sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, dml::targetToQ({ 0.0f, 2.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }), { 0.0f, 2.0f, 0.0f });

		//createObject("models/chess.glb", { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
		createLight({ -2.0f, 0.0f, -4.0f }, { 1.0f, 1.0f, 1.0f }, 1.0f, { 0.0f, 1.1f, 0.0f });
		createLight({ -2.0f, 0.0f, 4.0f }, { 1.0f, 1.0f, 1.0f }, 1.0f, { 0.0f, 0.7f, 0.0f });

		for (auto& obj : objects) {
			originalObjects.push_back(std::make_unique<model>(*obj));
		}

		setPlayer(1);
		setPlayer(2);
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
			if (isDeviceSuitableG(device) && isDeviceSuitableP(device, surface)) {
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
			throw std::runtime_error("failed to find a suitable GPU for graphics and presentation");
		}

		// use the best device
		physicalDevice = bestDevice;
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		printCapabilities(deviceProperties);

		// check if ray tracing is supported
		rtSupported = isRTSupported(physicalDevice);
		std::cout << "Raytacing is " << (rtSupported ? "supported" : "not supported") << "!!!!" << std::endl;
		sep();
	}

	void printCapabilities(VkPhysicalDeviceProperties deviceProperties) {
		sep();
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
		sep();
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamiliesG(physicalDevice);
		float queuePriority = 1.0f;
		VkDeviceQueueCreateInfo queueInf{};
		queueInf.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; //creates a structure to hold queue family info
		queueInf.queueFamilyIndex = indices.graphicsFamily.value(); // index of the queue family to create gotten from the findQueueFamilies function
		queueInf.queueCount = 1;
		queueInf.pQueuePriorities = &queuePriority;

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.imageCubeArray = VK_TRUE;
		VkPhysicalDeviceDescriptorIndexingFeatures descIndexing{};
		descIndexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
		descIndexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
		descIndexing.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
		descIndexing.runtimeDescriptorArray = VK_TRUE;
		descIndexing.descriptorBindingVariableDescriptorCount = VK_TRUE;
		descIndexing.descriptorBindingPartiallyBound = VK_TRUE;

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
		VK_KHR_MAINTENANCE3_EXTENSION_NAME
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
				throw std::runtime_error("Device contains unsupported extensions!");
			}
		}

		newInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		newInfo.ppEnabledExtensionNames = deviceExtensions.data();
		newInfo.enabledLayerCount = 0;
		newInfo.ppEnabledLayerNames = nullptr;
		VkResult result = vkCreateDevice(physicalDevice, &newInfo, nullptr, &device);
		if (result != VK_SUCCESS) {
			std::stringstream errorMessage;
			errorMessage << "Failed to create logical device! VkResult: " << result;
			throw std::runtime_error(errorMessage.str());
		}
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

	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool graphicsComplete() {
			return graphicsFamily.has_value();
		}
		bool presentComplete() {
			return presentFamily.has_value();
		}
	};

	QueueFamilyIndices findQueueFamiliesG(VkPhysicalDevice device) {
		QueueFamilyIndices indices;
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr); //this function gets the number of queue families
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data()); //outputs the queue families into the queueFamilies vector
		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) { //check if the queue family supports graphics
				indices.graphicsFamily = i;
			}
			if (indices.graphicsComplete()) {
				break;
			}
			i++;
		}

		return indices; //return the indices/position of the queue family that supports graphics
	}

	bool isDeviceSuitableG(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamiliesG(device);
		return indices.graphicsComplete(); //checks if the quefamilies have all been searched and if the graphics family has been found
	}
	QueueFamilyIndices findQueueFamiliesP(VkPhysicalDevice device, VkSurfaceKHR surface) {
		QueueFamilyIndices indices;
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (presentSupport) { // check if the queue family supports presentation
				indices.presentFamily = i;
			}
			if (indices.presentComplete()) {
				break;
			}
			i++;
		}
		return indices; //return the indices/position of the queue family that supports presentation
	}

	bool isDeviceSuitableP(VkPhysicalDevice device, VkSurfaceKHR surface) {
		QueueFamilyIndices indices = findQueueFamiliesP(device, surface);
		return indices.presentComplete(); //checks if the queue families have all been searched and if the present family has been found
	}

	struct SCsupportDetails { // struct to hold the swap chain details
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};
	SCsupportDetails querySCsupport(VkPhysicalDevice device) { //takes in the physical device and outputs the swap chain details
		SCsupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities); //get the surface capabilities. an example of a surface capability is the minimum and maximum number of images in the swap chain
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr); //get the number of formats. an example of a format is the pixel format and color space
		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr); //gets the number of present modes. this is the conditions for "swapping" images to the screen
		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}
		return details; //return the swap chain details
	}
	void initQueues() {
		QueueFamilyIndices indicesG = findQueueFamiliesG(physicalDevice);
		QueueFamilyIndices indicesP = findQueueFamiliesP(physicalDevice, surface);
		vkGetDeviceQueue(device, indicesG.graphicsFamily.value(), 0, &graphicsQueue); //params: device, queue family index, queue index, pointer to queue
		vkGetDeviceQueue(device, indicesP.presentFamily.value(), 0, &presentQueue);
	}
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		VkExtent2D actualExtent = { SCREEN_WIDTH, SCREEN_HEIGHT }; //extent=res
		actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));  //clamp the width between the min and max extents
		actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

		return actualExtent; //return the actual extent
	}
	void createSC() { //SC = swap chain
		SCsupportDetails swapChainSupport = querySCsupport(physicalDevice); //get the swap chain details from functions above
		// choose the best surface format, present mode, and swap extent for the swap chain.
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats); //paramiters datatype ism a VK surface format
		VkPresentModeKHR present = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
		swap.imageCount = swapChainSupport.capabilities.minImageCount + 1; //the number of images is based on the minimum number of images plus one
		if (swapChainSupport.capabilities.maxImageCount > 0 && swap.imageCount > swapChainSupport.capabilities.maxImageCount) {
			swap.imageCount = swapChainSupport.capabilities.maxImageCount;
		}
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
		QueueFamilyIndices indices = findQueueFamiliesG(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value() }; //the queue family indices that will be used
		newinfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; //image is owned by one queue family at a time and ownership must be explicitly transfered before using it in another queue family
		newinfo.queueFamilyIndexCount = 1;
		newinfo.pQueueFamilyIndices = queueFamilyIndices;
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
		createSCImageViews();
	}
	VkSurfaceFormatKHR  chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) { //choose the best surface format for the swap chain
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}
		return availableFormats[0];

	}
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
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
		throw std::runtime_error("failed to find suitable depth format! :(");
	}

	auto getAttributeIt(const std::string& name, const auto& attributes) {
		auto it = attributes.find(name);
		if (it == attributes.end()) {
			std::cerr << "WARNING: Failed to find attribute: " << name << std::endl;;
		}
		return it;
	}

	const float* getAccessorData(const auto& model, const auto& attributes, const std::string& attributeName) {
		auto it = getAttributeIt(attributeName, attributes); // get the attribute iterator
		const auto& accessor = model.accessors[it->second]; // get the accessor

		const auto& bufferView = model.bufferViews[accessor.bufferView]; // get the buffer view from the accessor
		const auto& buffer = model.buffers[bufferView.buffer]; // from the buffer view, get the buffer

		// return the data from the buffer
		// the data is the buffer data + the byte offset of the buffer view + the byte offset of the accessor!!
		return reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
	}

	const void* getIndexData(const auto& model, const auto& accessor) { // const void for flexablity
		const auto& bufferView = model.bufferViews[accessor.bufferView];
		const auto& buffer = model.buffers[bufferView.buffer];

		switch (accessor.componentType) {
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
			return reinterpret_cast<const uint8_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
			return reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
			return reinterpret_cast<const uint32_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
		default:
			std::cerr << "WARNING: Unsupported index type: " << accessor.componentType << std::endl;
			return nullptr;
		}
	}

	dml::mat4 calcNodeLM(const tinygltf::Node& node) { // get the local matrix of the node
		if (node.matrix.size() == 16) { // if the node already has a matrix just return it
			return dml::gltfToMat4(node.matrix);
		}

		// default values
		dml::vec3 t = { 0.0f, 0.0f, 0.0f };
		dml::vec4 r = { 0.0f, 0.0f, 0.0f, 1.0f };
		dml::vec3 s = { 1.0f, 1.0f, 1.0f };
		if (node.translation.size() >= 3) {
			t = {
				static_cast<float>(node.translation[0]),
				static_cast<float>(node.translation[1]),
				static_cast<float>(node.translation[2])
			};
		}

		if (node.rotation.size() >= 4) {
			r = {
				static_cast<float>(node.rotation[0]),
				static_cast<float>(node.rotation[1]),
				static_cast<float>(node.rotation[2]),
				static_cast<float>(node.rotation[3])
			};
		}

		if (node.scale.size() >= 3) {
			s = {
				static_cast<float>(node.scale[0]),
				static_cast<float>(node.scale[1]),
				static_cast<float>(node.scale[2])
			};
		}
		// calculate the matricies
		dml::mat4 translationMatrix = dml::translate(t);
		dml::mat4 rotationMatrix = dml::rotateQ(r); // quaternion rotation
		dml::mat4 scaleMatrix = dml::scale(s);
		return translationMatrix * rotationMatrix * scaleMatrix;
	}

	int getNodeIndex(const tinygltf::Model& model, int meshIndex) {
		for (size_t i = 0; i < model.nodes.size(); ++i) {
			if (model.nodes[i].mesh == meshIndex) {
				return static_cast<int>(i);
			}
		}
		return -1; // not found
	}

	dml::mat4 calcMeshWM(const tinygltf::Model& gltfMod, int meshIndex, std::unordered_map<int, int>& parentIndex, model& m) {
		int currentNodeIndex = getNodeIndex(gltfMod, meshIndex);
		dml::mat4 modelMatrix;

		// get the matricies for object positioning
		dml::mat4 translationMatrix = dml::translate(m.position);
		dml::mat4 rotationMatrix = dml::rotateQ(m.rotation);
		dml::mat4 scaleMatrix = dml::scale(m.scale * 0.03f); // 0.03 scales it down to a reasonable size

		// walk up the node hierarchy to accumulate transformations
		while (currentNodeIndex != -1) {
			const tinygltf::Node& node = gltfMod.nodes[currentNodeIndex];
			dml::mat4 localMatrix = calcNodeLM(node);

			// combine the localMatrix with the accumulated modelMatrix
			modelMatrix = localMatrix * modelMatrix;

			// move up to the parent node for the next iteration
			if (parentIndex.find(currentNodeIndex) != parentIndex.end()) {
				currentNodeIndex = parentIndex[currentNodeIndex];
			}
			else {
				currentNodeIndex = -1;  // no parent, exit loop
			}
		}

		// after accumulating all local matricies, apply scaling first and then translation and rotation
		modelMatrix = scaleMatrix * modelMatrix;
		modelMatrix = translationMatrix * rotationMatrix * modelMatrix;
		return modelMatrix;
	}

	void printNodeHierarchy(const tinygltf::Model& model, int nodeIndex, int depth = 0) {
		for (int i = 0; i < depth; ++i) { // indent based on depth
			std::cout << "  ";
		}
		// print the current node's name or index if the name is empty
		std::cout << "Node: " << (model.nodes[nodeIndex].name.empty() ? std::to_string(nodeIndex) : model.nodes[nodeIndex].name) << " matricies: " << model.nodes[nodeIndex].matrix.size() << std::endl;

		for (const auto& childIndex : model.nodes[nodeIndex].children) {
			printNodeHierarchy(model, childIndex, depth + 1);
		}
	}

	void printFullHierarchy(const tinygltf::Model& model) {
		std::unordered_set<int> childNodes;
		for (const auto& node : model.nodes) {
			for (const auto& childIndex : node.children) {
				childNodes.insert(childIndex);
			}
		}

		for (int i = 0; i < model.nodes.size(); ++i) {
			if (childNodes.find(i) == childNodes.end()) { // if a node doesn't appear in the childNodes set, it's a root
				printNodeHierarchy(model, i);
			}
		}
	}

	void loadSkybox(std::string path) {
		skybox.cubemap.path = SKYBOX_DIR + path;
		createTexturedCubemap(skybox.cubemap);
		createTextureImgView(skybox.cubemap, false, "cube");
		createTS(skybox.cubemap, false, "cube");

		skybox.bufferData.vertexOffset = 0;
		skybox.bufferData.vertexCount = 8;
		skybox.bufferData.indexOffset = 0;
		skybox.bufferData.indexCount = 36;
	}

	void loadModel(dml::vec3 scale, dml::vec3 pos, dml::vec4 rot, std::string path) {
		tf::Executor executor;
		tf::Taskflow taskFlow;
		uint32_t meshInd = 0; // index of the mesh in the model

		tinygltf::Model gltfModel;
		tinygltf::TinyGLTF loader;
		std::string err;
		std::string warn;

		bool ret = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, path);
		sep();
		std::cout << "Finished loading binaries" << std::endl;

		if (!warn.empty()) {
			std::cout << "Warning: " << warn << std::endl;
		}
		if (!err.empty()) {
			throw std::runtime_error(err);
		}
		if (!ret) {
			throw std::runtime_error("Failed to load GLTF model");
		}

		// get the index of the parent node for each node
		std::unordered_map<int, int> parentInd;
		for (size_t nodeIndex = 0; nodeIndex < gltfModel.nodes.size(); ++nodeIndex) {
			const auto& node = gltfModel.nodes[nodeIndex];
			for (const auto& childIndex : node.children) {
				parentInd[childIndex] = static_cast<int>(nodeIndex);
			}
		}

		// check if the model has any skins or animations (not supported for now)
		if (!gltfModel.skins.empty()) {
			std::cerr << "WARNING: The " << path << " contains skinning information" << std::endl;
		}

		if (!gltfModel.animations.empty()) {
			std::cerr << "WARNING: The " << path << " contains animation data." << std::endl;
		}

		// check if the gltf model relies on any extensions
		for (const auto& extension : gltfModel.extensionsUsed) {
			std::cerr << "WARNING: The " << path << " relies on: " << extension << std::endl;
		}
		//printFullHierarchy(gltfModel);

		// parallel loading using taskflow:
		auto loadModelTask = taskFlow.emplace([&]() {
			// loop over each mesh (object)
			for (const auto& mesh : gltfModel.meshes) {
				model newObject;

				std::unordered_map<dvl::Vertex, uint32_t, dvl::vertHash> uniqueVertices;
				std::vector<dvl::Vertex> tempVertices;
				std::vector<uint32_t> tempIndices;


				// process primitives in the mesh
				for (const auto& primitive : mesh.primitives) {
					if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
						std::cerr << "WARNING: Unsupported primitive mode: " << primitive.mode << std::endl;
					}
					bool tangentFound = true;
					bool colorFound = true;

					// pos
					auto positionIt = getAttributeIt("POSITION", primitive.attributes);
					const auto& positionAccessor = gltfModel.accessors[positionIt->second];
					const float* positionData = getAccessorData(gltfModel, primitive.attributes, "POSITION");

					// tex coords
					auto texCoordIt = getAttributeIt("TEXCOORD_0", primitive.attributes);
					const auto& texCoordAccessor = gltfModel.accessors[texCoordIt->second];
					const float* texCoordData = getAccessorData(gltfModel, primitive.attributes, "TEXCOORD_0");

					// normals
					auto normalIt = getAttributeIt("NORMAL", primitive.attributes);
					const auto& normalAccessor = gltfModel.accessors[normalIt->second];
					const float* normalData = getAccessorData(gltfModel, primitive.attributes, "NORMAL");

					// colors
					const float* colorData = nullptr;
					auto colorIt = getAttributeIt("COLOR_0", primitive.attributes);
					if (colorIt != primitive.attributes.end()) { // check if the primitive has color data
						const auto& colorAccessor = gltfModel.accessors[colorIt->second];
						colorData = getAccessorData(gltfModel, primitive.attributes, "COLOR_0");
					}
					else {
						colorFound = false;
					}

					// indices
					const auto& indexAccessor = gltfModel.accessors[primitive.indices];
					const void* rawIndices = getIndexData(gltfModel, indexAccessor);

					// tangents
					std::vector<dml::vec4> tangents(positionAccessor.count, dml::vec4{ 0.0f, 0.0f, 0.0f, 0.0f });

					const float* tangentData = nullptr;
					auto tangentIt = getAttributeIt("TANGENT", primitive.attributes);
					if (tangentIt != primitive.attributes.end()) { // check if the primitive has tangents
						const auto& tangentAccessor = gltfModel.accessors[tangentIt->second];
						tangentData = getAccessorData(gltfModel, primitive.attributes, "TANGENT");
					}
					else {
						std::cout << "Calculating tangents..." << std::endl;
						tangentFound = false;

						switch (indexAccessor.componentType) {
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
							dvl::calculateTangents<uint8_t>(positionData, texCoordData, tangents, rawIndices, indexAccessor.count);
							break;
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
							dvl::calculateTangents<uint16_t>(positionData, texCoordData, tangents, rawIndices, indexAccessor.count);
							break;
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
							dvl::calculateTangents<uint32_t>(positionData, texCoordData, tangents, rawIndices, indexAccessor.count);
							break;
						default:
							std::cerr << "WARNING: Unsupported index type: " << indexAccessor.componentType << std::endl;
							break;
						}

						dvl::normalizeTangents(tangents);
					}

					for (size_t i = 0; i < indexAccessor.count; ++i) {
						uint32_t index;  // use the largest type to ensure no overflow.

						switch (indexAccessor.componentType) {
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
							index = static_cast<const uint8_t*>(rawIndices)[i];
							break;
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
							index = static_cast<const uint16_t*>(rawIndices)[i];
							break;
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
							index = static_cast<const uint32_t*>(rawIndices)[i];
							break;
						default:
							continue; // skip this iteration
						}

						dvl::Vertex vertex;
						vertex.pos = { positionData[3 * index], positionData[3 * index + 1], positionData[3 * index + 2] };
						vertex.tex = { texCoordData[2 * index], texCoordData[2 * index + 1] };
						vertex.normal = { normalData[3 * index], normalData[3 * index + 1], normalData[3 * index + 2] };

						if (colorFound) {
							vertex.col = { colorData[4 * index], colorData[4 * index + 1], colorData[4 * index + 2], colorData[4 * index + 3] };
						}
						else {
							vertex.col = { 1.0f, 1.0f, 1.0f, 1.0f };
						}
						//vertex.col.w = 0.6f;

						// get handedness of the tangent
						dml::vec3 t = tangents[index].xyz();
						tangents[index].w = dml::dot(dml::cross(vertex.normal, t), tangents[index].xyz()) < 0.0f ? -1.0f : 1.0f;

						if (tangentFound) {
							vertex.tangent = { tangentData[4 * index], tangentData[4 * index + 1], tangentData[4 * index + 2], tangentData[4 * index + 3] };
							//std::cout << "calculated tangent: " << tangents[index] << "other tangent: " << forms::vec4(tangentData[4 * index], tangentData[4 * index + 1], tangentData[4 * index + 2], tangentData[4 * index + 3]) << std::endl;
						}
						else {
							vertex.tangent = tangents[index];

						}

						if (uniqueVertices.count(vertex) == 0) {
							uniqueVertices[vertex] = static_cast<uint32_t>(tempVertices.size());
							tempVertices.push_back(std::move(vertex));
						}
						tempIndices.push_back(uniqueVertices[vertex]);
					}
					if (primitive.material >= 0) { // if the primitive has a material
						auto& material = gltfModel.materials[primitive.material];
						Materials texture;

						// base color texture
						if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
							auto& texInfo = material.pbrMetallicRoughness.baseColorTexture;
							auto& tex = gltfModel.textures[texInfo.index];
							texture.baseColor.gltfImage = gltfModel.images[tex.source];
							texture.baseColor.path = "gltf";
							texture.baseColor.found = true;
							newObject.textureCount++;
						}

						// metallic-roughness Texture
						if (material.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
							auto& texInfo = material.pbrMetallicRoughness.metallicRoughnessTexture;
							auto& tex = gltfModel.textures[texInfo.index];
							texture.metallicRoughness.gltfImage = gltfModel.images[tex.source];
							texture.metallicRoughness.path = "gltf";
							texture.metallicRoughness.found = true;
							newObject.textureCount++;
						}

						// normal map
						if (material.normalTexture.index >= 0) {
							auto& texInfo = material.normalTexture;
							auto& tex = gltfModel.textures[texInfo.index];
							texture.normalMap.gltfImage = gltfModel.images[tex.source];
							texture.normalMap.path = "gltf";
							texture.normalMap.found = true;
							newObject.textureCount++;
						}

						// emissive map
						if (material.emissiveTexture.index >= 0) {
							auto& texInfo = material.emissiveTexture;
							auto& tex = gltfModel.textures[texInfo.index];
							texture.emissiveMap.gltfImage = gltfModel.images[tex.source];
							texture.emissiveMap.path = "gltf";
							texture.emissiveMap.found = true;
							newObject.textureCount++;
						}

						// occlusion map
						if (material.occlusionTexture.index >= 0) {
							auto& texInfo = material.occlusionTexture;
							auto& tex = gltfModel.textures[texInfo.index];
							texture.occlusionMap.gltfImage = gltfModel.images[tex.source];
							texture.occlusionMap.path = "gltf";
							texture.occlusionMap.found = true;
							newObject.textureCount++;
						}

						// ensure the model is PBR
						if (!texture.baseColor.found && !texture.metallicRoughness.found && !texture.normalMap.found && !texture.emissiveMap.found && !texture.occlusionMap.found) {
							std::cerr << "WARNING: Model isnt PBR!!" << std::endl;
							return;
						}

						newObject.material = texture;
					}

					else {
						std::cerr << "WARNING: Primitive " << primitive.material << " doesn't have a material/texture" << std::endl;
					}
				}
				newObject.vertices = tempVertices;
				newObject.indices = tempIndices;

				size_t hash1 = std::hash<std::size_t>{}(meshInd * tempIndices.size() * tempVertices.size());
				size_t hash2 = std::hash<std::string>{}(mesh.name);

				newObject.modelHash = hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));

				newObject.name = mesh.name;

				newObject.scale = scale;
				newObject.position = pos;
				newObject.rotation = rot;

				// calculate the model matrix for the mesh
				newObject.modelMatrix = calcMeshWM(gltfModel, meshInd, parentInd, newObject);

				// add newObject to global objects list
				modelMtx.lock();
				objects.push_back(std::make_unique<model>(newObject));
				modelMtx.unlock();

				modelIndex++;
				meshInd++;
			}
			std::cout << "Finished loading vertecies" << std::endl;
			}).name("load_model");

			auto loadTextureTask = taskFlow.emplace([&]() {
				for (auto& object : objects) {
					//create the texture image for each texture (material)
					//also create mipmaps for every texture
					if (object->material.baseColor.found) {
						createTexturedImage(object->material.baseColor, true);
						createTextureImgView(object->material.baseColor, true);
						createTS(object->material.baseColor, true);
					}

					if (object->material.metallicRoughness.found) {
						createTexturedImage(object->material.metallicRoughness, true, "metallic");
						createTextureImgView(object->material.metallicRoughness, true, "metallic");
						createTS(object->material.metallicRoughness, true, "metallic");
					}

					if (object->material.normalMap.found) {
						createTexturedImage(object->material.normalMap, true, "norm");
						createTextureImgView(object->material.normalMap, true, "norm");
						createTS(object->material.normalMap, true, "norm");
					}

					if (object->material.emissiveMap.found) {
						createTexturedImage(object->material.emissiveMap, true, "emissive");
						createTextureImgView(object->material.emissiveMap, true, "emissive");
						createTS(object->material.emissiveMap, true, "emissive");
					}

					if (object->material.occlusionMap.found) {
						createTexturedImage(object->material.occlusionMap, true, "occlusion");
						createTextureImgView(object->material.occlusionMap, true, "occlusion");
						createTS(object->material.occlusionMap, true, "occlusion");
					}
				}
				}).name("load_texture");
				loadModelTask.precede(loadTextureTask);
				executor.run(taskFlow).get();

				sep();
				std::cout << "Successfully loaded " << objects.size() << " meshes" << std::endl;
				taskFlow.clear();
	}

	void setupTextures() {
		depthFormat = findDepthFormat();

		// main pass color image
		createImage(mainPassTextures.color.image, mainPassTextures.color.memory, swap.extent.width, swap.extent.height, swap.imageFormat, 1, 1, false, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		createTextureImgView(mainPassTextures.color, false, "swap");
		createTS(mainPassTextures.color, false, "swap");
		transitionImageLayout(mainPassTextures.color.image, swap.imageFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, 0);

		// main pass depth image
		createImage(mainPassTextures.depth.image, mainPassTextures.depth.memory, swap.extent.width, swap.extent.height, depthFormat, 1, 1, false, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		createTextureImgView(mainPassTextures.depth, false, "depth");
		createTS(mainPassTextures.depth, false, "depth");

		// weighted color image
		createImage(wboit.weightedColor.image, wboit.weightedColor.memory, swap.extent.width, swap.extent.height, swap.imageFormat, 1, 1, false, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		createTextureImgView(wboit.weightedColor, false, "swap");
		createTS(wboit.weightedColor, false, "swap");

		// weighted alpha image
		createImage(wboit.weightedAlpha.image, wboit.weightedAlpha.memory, swap.extent.width, swap.extent.height, swap.imageFormat, 1, 1, false, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		createTextureImgView(wboit.weightedAlpha, false, "swap");
		createTS(wboit.weightedAlpha, false, "swap");

		// skybox image (2d)
		createImage(skybox.out.image, skybox.out.memory, swap.extent.width, swap.extent.height, swap.imageFormat, 1, 1, false, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		createTextureImgView(skybox.out, false, "swap");
		createTS(skybox.out, false, "swap");
	}

	void setupShadowMaps() { // initialize the shadow maps for each light
		for (size_t i = 0; i < lights.size(); i++) {
			createImage(lights[i]->shadowMapData.image, lights[i]->shadowMapData.memory, shadowProps.mapWidth, shadowProps.mapHeight, depthFormat, 1, 1, false, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
			createTextureImgView(lights[i]->shadowMapData, false, "depth");
			createTS(lights[i]->shadowMapData, false, "depth");
		}
	}

	void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t layerCount, uint32_t levelCount, uint32_t baseMip) {
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		if (format == VK_FORMAT_D32_SFLOAT) { // if the format is a depth format
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		}
		else {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		barrier.subresourceRange.baseMipLevel = baseMip;
		barrier.subresourceRange.levelCount = levelCount;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = layerCount;

		// earliest stage in the pipeline that will wait on the barrier to be passed
		VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
				sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			}
			else {
				barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
				destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			}
		}
		else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
				destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			}
			else {
				barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
				destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			}
		}
		else if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		}
		else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else {
			throw std::invalid_argument("Unsupported layout transition!");
		}
		vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier); // insert the barrier into the command buffer
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t layerCount, uint32_t levelCount, uint32_t baseMip) {
		VkCommandBuffer tempCommandBuffer = beginSingleTimeCommands(commandPool);
		transitionImageLayout(tempCommandBuffer, image, format, oldLayout, newLayout, layerCount, levelCount, baseMip);
		endSingleTimeCommands(tempCommandBuffer, commandPool);
	}

	void createImage(VkImage& image, VkDeviceMemory& imageMemory, uint32_t width, uint32_t height, VkFormat format, uint32_t mipLevels, uint32_t arrayLayers, bool cubeMap, VkImageUsageFlags usage) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.arrayLayers = arrayLayers;
		imageInfo.mipLevels = mipLevels;
		imageInfo.format = format;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (cubeMap) imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create color image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate color image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
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

	void setupModelMatInstanceBuffer() {
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = sizeof(modelMatInstanceData);
		bufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &instanceBuffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create instance buffer!");
		}

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(device, instanceBuffer, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (vkAllocateMemory(device, &allocateInfo, nullptr, &instanceBufferMem) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate memory for the instance buffer!");
		}

		vkBindBufferMemory(device, instanceBuffer, instanceBufferMem, 0);

		// once memory is bound, map and fill it
		void* data;
		vkMapMemory(device, instanceBufferMem, 0, bufferCreateInfo.size, 0, &data);
		memcpy(data, &objInstanceData, bufferCreateInfo.size);
		vkUnmapMemory(device, instanceBufferMem);
	}


	void setupCamMatUBO() { // ubo containing the cameras matricies (view and projection)
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = sizeof(camUBO);
		bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // will only be used by one queue family

		if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &cam.buffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create matrix SSBO for camera!");
		}

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(device, cam.buffer, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (vkAllocateMemory(device, &allocateInfo, nullptr, &cam.bufferMem) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate memory for the matrix SSBO!");
		}

		vkBindBufferMemory(device, cam.buffer, cam.bufferMem, 0);

		// once memory is bound, map and fill it
		void* data;
		vkMapMemory(device, cam.bufferMem, 0, bufferCreateInfo.size, 0, &data);
		memcpy(data, &camMatData, bufferCreateInfo.size);
		vkUnmapMemory(device, cam.bufferMem);
	}

	void setupLights() {
		VkBufferCreateInfo bufferCreateInfo = {};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = sizeof(lightDataSSBO);
		bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		vkCreateBuffer(device, &bufferCreateInfo, nullptr, &lightBuffer);

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, lightBuffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		vkAllocateMemory(device, &allocInfo, nullptr, &lightBufferMem);

		vkBindBufferMemory(device, lightBuffer, lightBufferMem, 0);

		void* data;
		vkMapMemory(device, lightBufferMem, 0, bufferCreateInfo.size, 0, &data);
		memcpy(data, &lightData, bufferCreateInfo.size);
		vkUnmapMemory(device, lightBufferMem);
	}


	void printVec3(const dml::vec3& vector) {
		std::cout << "{" << vector.x << ", " << vector.y << ", " << vector.z << "}" << std::endl;
	}

	void printVec4(const dml::vec4& vector) {
		std::cout << "{" << vector.x << ", " << vector.y << ", " << vector.z << ", " << vector.w << "}" << std::endl;
	}

	void calcCameraMats() {
		cam.viewMatrix = cam.getViewMatrix();
		cam.projectionMatrix = dml::projection(60.0f, swap.extent.width / static_cast<float>(swap.extent.height), 0.1f, 1000.0f);

	}

	void calcShadowMats(light& l) {
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
			light& l = *lights[i];
			calcShadowMats(l);
			memcpy(&lightData.lightsMatricies[i].proj, &lights[i]->proj, sizeof(lights[i]->proj));
			memcpy(&lightData.lightsMatricies[i].view, &lights[i]->view, sizeof(lights[i]->view));
			copyLightToLightCords(l, lightData.lightCords[i]);
		}
		void* lData;
		vkMapMemory(device, lightBufferMem, 0, sizeof(lightData), 0, &lData);
		memcpy(lData, &lightData, sizeof(lightData));
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

	void copyLightToLightCords(const light& src, lightCords& dest) {
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

	std::vector<Texture> getAllTextures() {
		allTextures.reserve(totalTextureCount);
		size_t currentIndex = 0;
		for (size_t i = 0; i < objects.size(); i++) {
			auto& obj = objects[i];
			if (uniqueModelIndex[obj->modelHash] == i) {
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
			if (uniqueModelIndex[obj->modelHash] == i) {
				objects[i]->texIndex = i;
			}
		}
		std::cout << "Finished loading " << totalTextureCount << " textures!" << std::endl;
		return allTextures;
	}

	std::vector<shadowMapDataObject> getAllShadowMaps() {
		std::vector<shadowMapDataObject>allMaps;
		allMaps.reserve(lights.size());
		for (const auto& light : lights) {
			allMaps.push_back(light->shadowMapData);
		}
		return allMaps;
	}

	VkDescriptorSetLayout createDSLayout(uint32_t bindingIndex, VkDescriptorType type, uint32_t descriptorCount, VkShaderStageFlags stageFlags) {
		VkDescriptorSetLayoutBinding binding{};
		binding.binding = bindingIndex;
		binding.descriptorType = type;
		binding.descriptorCount = descriptorCount;
		binding.stageFlags = stageFlags;

		VkDescriptorBindingFlagsEXT bindingFlags = 0;
		//if the descriptor count is variable, set the flag
		if (descriptorCount > 1) {
			bindingFlags |= VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT;
		}

		VkDescriptorSetLayoutBindingFlagsCreateInfoEXT bindingFlagsInfo{};
		bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
		bindingFlagsInfo.bindingCount = 1;
		bindingFlagsInfo.pBindingFlags = &bindingFlags; // if 0, no flags are set

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.pNext = &bindingFlagsInfo; // bindingFlagsInfo is added to the pNext chain
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &binding;

		VkDescriptorSetLayout descriptorSetLayout;
		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor set layout!");
		}

		return descriptorSetLayout;
	}

	VkDescriptorPool createDSPool(VkDescriptorType type, uint32_t descriptorCount) {
		VkDescriptorPoolSize poolSize{};
		poolSize.type = type;
		poolSize.descriptorCount = descriptorCount;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		poolInfo.maxSets = 1;

		VkDescriptorPool descriptorPool;
		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor pool!");
		}

		return descriptorPool;
	}

	void createDS() {
		const uint8_t size = 7;
		descs.sets.resize(size);
		descs.layouts.resize(size);
		descs.pools.resize(size);
		uint32_t lightSize = static_cast<uint32_t>(lights.size());

		//initialize descriptor set layouts and pools
		descs.layouts[0] = createDSLayout(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(totalTextureCount), VK_SHADER_STAGE_FRAGMENT_BIT); // array of textures
		descs.layouts[1] = createDSLayout(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT); // light data ssbo
		descs.layouts[2] = createDSLayout(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, lightSize, VK_SHADER_STAGE_FRAGMENT_BIT); // array of shadow map samplers
		descs.layouts[3] = createDSLayout(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // 1 sampler for the skybox
		descs.layouts[4] = createDSLayout(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT); // camera matricies ubo
		descs.layouts[5] = createDSLayout(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, VK_SHADER_STAGE_FRAGMENT_BIT); // textures for composition pass
		descs.layouts[6] = createDSLayout(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // texture for main pass depth

		descs.pools[0] = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(totalTextureCount));
		descs.pools[1] = createDSPool(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
		descs.pools[2] = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, lightSize);
		descs.pools[3] = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1); // skybox
		descs.pools[4] = createDSPool(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);
		descs.pools[5] = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4);
		descs.pools[6] = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1);

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorSetCount = 1;

		std::vector<uint32_t> descCountArr = { static_cast<uint32_t>(totalTextureCount), 1, lightSize, 1, 1, 4, 1 };

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

		std::vector<shadowMapDataObject> shadowMaps = getAllShadowMaps(); // put all shadowmaps into 1 vector

		VkDescriptorBufferInfo camMatBufferInfo{};
		camMatBufferInfo.buffer = cam.buffer;
		camMatBufferInfo.offset = 0;
		camMatBufferInfo.range = sizeof(camUBO);

		std::vector<VkDescriptorImageInfo> imageInfos(totalTextureCount);
		for (size_t i = 0; i < totalTextureCount; i++) {
			imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfos[i].imageView = allTextures[i].imageView;
			imageInfos[i].sampler = allTextures[i].sampler;
		}
		std::vector<VkDescriptorImageInfo> shadowInfos(lights.size());
		for (size_t i = 0; i < lights.size(); i++) {
			shadowInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			shadowInfos[i].imageView = shadowMaps[i].imageView;
			shadowInfos[i].sampler = shadowMaps[i].sampler;
		}

		std::array<VkDescriptorImageInfo, 4> compositionPassImageInfo{};
		compositionPassImageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		compositionPassImageInfo[0].imageView = mainPassTextures.color.imageView;
		compositionPassImageInfo[0].sampler = mainPassTextures.color.sampler;

		compositionPassImageInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		compositionPassImageInfo[1].imageView = wboit.weightedColor.imageView;
		compositionPassImageInfo[1].sampler = wboit.weightedColor.sampler;

		compositionPassImageInfo[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		compositionPassImageInfo[2].imageView = wboit.weightedAlpha.imageView;
		compositionPassImageInfo[2].sampler = wboit.weightedAlpha.sampler;

		compositionPassImageInfo[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		compositionPassImageInfo[3].imageView = skybox.out.imageView;
		compositionPassImageInfo[3].sampler = skybox.out.sampler;

		VkDescriptorImageInfo mainPassDepthInfo{};
		mainPassDepthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		mainPassDepthInfo.imageView = mainPassTextures.depth.imageView;
		mainPassDepthInfo.sampler = mainPassTextures.depth.sampler;

		VkDescriptorImageInfo skyboxInfo{};
		skyboxInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		skyboxInfo.imageView = skybox.cubemap.imageView;
		skyboxInfo.sampler = skybox.cubemap.sampler;

		VkDescriptorBufferInfo lightBufferInfo{};
		lightBufferInfo.buffer = lightBuffer;
		lightBufferInfo.offset = 0;
		lightBufferInfo.range = sizeof(lightDataSSBO);

		std::array<VkWriteDescriptorSet, size> descriptorWrites{}; // vector to hold the info about the UBO and the texture sampler

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descs.sets[0];
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[0].descriptorCount = static_cast<uint32_t>(totalTextureCount);
		descriptorWrites[0].pImageInfo = imageInfos.data();

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descs.sets[1];
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;//type=SSBO
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &lightBufferInfo;

		descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[2].dstSet = descs.sets[2];
		descriptorWrites[2].dstBinding = 2;
		descriptorWrites[2].dstArrayElement = 0;
		descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[2].descriptorCount = static_cast<uint32_t>(lights.size());
		descriptorWrites[2].pImageInfo = shadowInfos.data();

		descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[3].dstSet = descs.sets[3];
		descriptorWrites[3].dstBinding = 3;
		descriptorWrites[3].dstArrayElement = 0;
		descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[3].descriptorCount = 1;
		descriptorWrites[3].pImageInfo = &skyboxInfo;

		descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[4].dstSet = descs.sets[4];
		descriptorWrites[4].dstBinding = 4;
		descriptorWrites[4].dstArrayElement = 0;
		descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; //type=ubo
		descriptorWrites[4].descriptorCount = 1;
		descriptorWrites[4].pBufferInfo = &camMatBufferInfo;

		descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[5].dstSet = descs.sets[5];
		descriptorWrites[5].dstBinding = 5;
		descriptorWrites[5].dstArrayElement = 0;
		descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[5].descriptorCount = 4;
		descriptorWrites[5].pImageInfo = compositionPassImageInfo.data();

		descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[6].dstSet = descs.sets[6];
		descriptorWrites[6].dstBinding = 6;
		descriptorWrites[6].dstArrayElement = 0;
		descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[6].descriptorCount = 1;
		descriptorWrites[6].pImageInfo = &mainPassDepthInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}

	void setupDescriptorSets(bool initial = true) {
		descs.sets.clear();
		totalTextureCount = 0;
		for (uint32_t i = 0; i < objects.size(); i++) {
			auto& obj = objects[i];
			if (uniqueModelIndex[obj->modelHash] == i) {
				totalTextureCount += obj->textureCount;
			}
		}
		if (initial) {
			getAllTextures();
			setupCamMatUBO(); //create the camera matrix UBO
			setupLights();
		}
		createDS(); //create the descriptor set
	}

	template<typename T>
	void createTS(T& tex, bool doMipmap, std::string type = "tex") {
		VkSamplerCreateInfo samplerInf{};
		samplerInf.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInf.magFilter = VK_FILTER_LINEAR; // magnification filter
		samplerInf.minFilter = VK_FILTER_LINEAR; // minification filter
		samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT; // repeat the texture when out of bounds (horizontal)
		samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT; // (vertical)
		samplerInf.anisotropyEnable = VK_FALSE; // warps textures to fit objects, etc
		samplerInf.maxAnisotropy = 16;
		samplerInf.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInf.unnormalizedCoordinates = VK_FALSE;
		if (type == "depth") {
			samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.compareEnable = VK_TRUE;
			samplerInf.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
			samplerInf.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		}
		if (type == "cube") {
			samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // prevent seams at the edges
		}
		samplerInf.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInf.minLod = 0.0f;
		samplerInf.maxLod = doMipmap ? static_cast<float>(tex.mipLevels) : 1;
		if (vkCreateSampler(device, &samplerInf, nullptr, &tex.sampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	template<typename T>
	void createTextureImgView(T& tex, bool doMipmap, std::string type = "base") {
		VkImageViewCreateInfo viewInf{};
		viewInf.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInf.image = tex.image;
		viewInf.subresourceRange.baseArrayLayer = 0;
		viewInf.subresourceRange.layerCount = 1;
		viewInf.viewType = VK_IMAGE_VIEW_TYPE_2D;
		if (type == "depth") {
			viewInf.format = VK_FORMAT_D32_SFLOAT;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		}
		else if (type == "norm") {
			viewInf.format = VK_FORMAT_R8G8B8A8_UNORM;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else if (type == "base") {
			viewInf.format = VK_FORMAT_R8G8B8A8_SRGB; // for base texture
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else if (type == "swap") {
			viewInf.format = swap.imageFormat; // format of the swap chain
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else if (type == "metallic") {
			viewInf.format = VK_FORMAT_R8G8B8A8_UNORM; // for metallic roughness
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else if (type == "emissive") {
			viewInf.format = VK_FORMAT_R8G8B8A8_UNORM;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else if (type == "occlusion") {
			viewInf.format = VK_FORMAT_R8G8B8A8_UNORM;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else if (type == "cube") {
			viewInf.format = VK_FORMAT_R32G32B32A32_SFLOAT; // for cubemaps
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInf.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
			viewInf.subresourceRange.layerCount = 6;
		}
		else if (type == "alpha") {
			viewInf.format = VK_FORMAT_R32_SFLOAT;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else {
			throw std::invalid_argument("Invalid texture type!");
		}
		viewInf.subresourceRange.baseMipLevel = 0;
		viewInf.subresourceRange.levelCount = doMipmap ? tex.mipLevels - viewInf.subresourceRange.baseMipLevel : 1; //miplevel is influenced by the base 
		if (vkCreateImageView(device, &viewInf, nullptr, &tex.imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}
	}

	void getGLTFImageData(const tinygltf::Image& gltfImage, Texture& t, unsigned char*& imgData) {
		int width = gltfImage.width; // Set the texture's width, height, and channels
		int height = gltfImage.height;
		int channels = gltfImage.component;


		// delete previously allocated memory if any
		if (imgData != nullptr) {
			delete[] imageData;
		}

		imgData = new unsigned char[width * height * 4]; // create a new array to hold the image data

		// iterate through the image data and copy it into the new array
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				for (int c = 0; c < channels; ++c) {
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

	void getImageDataHDR(std::string path, Texture& t, float*& imgData) {
		int texWidth, texHeight, texChannels;
		imgData = stbi_loadf(path.c_str(), &texWidth, &texHeight, &texChannels, 4); // load RGBA (alpha not used) for the R32G32B32A32_SFLOAT format
		t.width = texWidth;
		t.height = texHeight;
		if (!imgData) {
			std::string error = stbi_failure_reason(); // get the detailed error
			throw std::runtime_error("failed to load HDR image: " + path + "! Reason: " + error);
		}
	}


	void createStagingBuffer(Texture& tex, bool cubeMap) { // buffer to transfer data from the CPU (imageData) to the GPU sta
		VkBufferCreateInfo bufferInf{};
		auto bpp = cubeMap ? sizeof(float) * 4 : 4;
		VkDeviceSize imageSize;
		if (cubeMap) {
			uint32_t faceWidth = tex.width / 4;
			uint32_t faceHeight = tex.height / 3;
			VkDeviceSize faceSize = static_cast<VkDeviceSize>(faceWidth) * faceHeight * bpp;
			imageSize = static_cast<VkDeviceSize>(tex.width) * tex.height * bpp;
		}
		else {
			imageSize = static_cast<VkDeviceSize>(tex.width) * tex.height * bpp;
		}
		bufferInf.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInf.size = imageSize;
		bufferInf.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufferInf.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (vkCreateBuffer(device, &bufferInf, nullptr, &tex.stagingBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create staging buffer!");
		}
		// get mem requirements;
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, tex.stagingBuffer, &memRequirements);
		VkMemoryAllocateInfo allocInf{};
		allocInf.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInf.allocationSize = memRequirements.size;
		allocInf.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (vkAllocateMemory(device, &allocInf, nullptr, &tex.stagingBufferMem) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate staging buffer memory!");
		}

		// bind the memory to the buffer:
		vkBindBufferMemory(device, tex.stagingBuffer, tex.stagingBufferMem, 0);
		void* data;
		if (vkMapMemory(device, tex.stagingBufferMem, 0, imageSize, 0, &data) != VK_SUCCESS) {
			throw std::runtime_error("failed to map staging buffer memory!");
		}
		if (cubeMap) {
			memcpy(data, skyboxData, imageSize);
		}
		else {
			memcpy(data, imageData, imageSize);
		}
		vkUnmapMemory(device, tex.stagingBufferMem);
	}

	void createTexturedCubemap(Texture& tex) {
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

		createImage(tex.image, tex.memory, faceWidth, faceHeight, VK_FORMAT_R32G32B32A32_SFLOAT, 1, 6, true, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

		transitionImageLayout(tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, 1, 0);
		VkCommandBuffer copyCmdBuffer = beginSingleTimeCommands(commandPool);

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
		endSingleTimeCommands(copyCmdBuffer, commandPool);

		transitionImageLayout(tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6, 1, 0);
		stbi_image_free(skyboxData);
		skyboxData = nullptr;
	}

	void createTexturedImage(Texture& tex, bool doMipmap, std::string type = "base") {
		if (tex.stagingBuffer == VK_NULL_HANDLE) {
			if (tex.path != "gltf") { // standard image
				getImageData(tex.path, imageData);
			}
			else {
				getGLTFImageData(tex.gltfImage, tex, imageData);
			}
			createStagingBuffer(tex, false);
			tex.mipLevels = doMipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(tex.width, tex.height)))) + 1 : 1;

			VkFormat imgFormat = type == "base" ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
			createImage(tex.image, tex.memory, tex.width, tex.height, imgFormat, tex.mipLevels, 1, false, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

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

			VkCommandBuffer tempBuffer = beginSingleTimeCommands(commandPool);

			transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, tex.mipLevels, 0);
			vkCmdCopyBufferToImage(tempBuffer, tex.stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region); //copy the data from the staging buffer to the image

			int mipWidth = tex.width;
			int mipHeight = tex.height;
			if (doMipmap) {
				for (uint32_t j = 0; j < tex.mipLevels; j++) {
					transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 1, j);

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

					transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, j);

					//for the next mip level, divide the width and height by 2, unless they are already 1
					if (mipWidth > 1) mipWidth /= 2;
					if (mipHeight > 1) mipHeight /= 2;
				}
			}
			else {
				transitionImageLayout(tempBuffer, tex.image, imgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, tex.mipLevels, 0);
			}

			endSingleTimeCommands(tempBuffer, commandPool);
			stbi_image_free(imageData);
			imageData = nullptr;
		}
	}

	VkCommandBuffer beginSingleTimeCommands(VkCommandPool cPool) {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; //specifies if the command buffer is primary or secondary
		allocInfo.commandPool = cPool;
		allocInfo.commandBufferCount = 1;
		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; //one time command buffer
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer cBuffer, VkCommandPool cPool) {
		vkEndCommandBuffer(cBuffer);
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cBuffer;
		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE); //submit the command buffer to the queue
		vkQueueWaitIdle(graphicsQueue); //wait for the queue to be idle
		vkFreeCommandBuffers(device, cPool, 1, &cBuffer); //free the command buffer
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
		instanceBindDesc.stride = sizeof(modelMat);
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
			attrDesc[index].offset = offsetof(modelMat, model) + sizeof(float) * 4 * i;
		}

		attrDesc[9].binding = 1;
		attrDesc[9].location = 9;
		attrDesc[9].format = VK_FORMAT_R32_UINT; // 1 uint32_t
		attrDesc[9].offset = offsetof(modelMat, render);

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
		vp.x = 0.0f;
		vp.y = 0.0f;
		vp.width = static_cast<float>(swap.extent.width); // swap chain extent width for the viewport
		vp.height = static_cast<float>(swap.extent.height);
		vp.minDepth = 0.0f;
		vp.maxDepth = 1.0f;
		VkRect2D scissor{};
		scissor.offset = { 0, 0 }; // top-left corner offset
		scissor.extent = swap.extent;
		VkPipelineViewportStateCreateInfo vpState{};
		vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		vpState.viewportCount = 1;
		vpState.pViewports = &vp;
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

		//dynamic state setup: Allows for the dynamic changing of state without having to recreate the pipeline
		VkDynamicState dynamicStates[] = {
			VK_DYNAMIC_STATE_VIEWPORT
		};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(std::size(dynamicStates));
		dynamicState.pDynamicStates = dynamicStates;

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
		pipelineInf.pDynamicState = &dynamicState;
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
		instanceBindDesc.stride = sizeof(modelMat);
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
			attrDesc[index].offset = offsetof(modelMat, model) + sizeof(float) * 4 * i;
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

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.attachmentCount = 0;

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int) * 2; // 2 ints for the light index and the objects model matrix index

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
		pipelineInfo.pColorBlendState = &colorBlending;
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
		vpState.pViewports = &vp;
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

		VkDynamicState dynamicStates[] = {
			VK_DYNAMIC_STATE_VIEWPORT
		};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(std::size(dynamicStates));
		dynamicState.pDynamicStates = dynamicStates;

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
		pipelineInf.pDynamicState = &dynamicState;
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
		instanceBindDesc.stride = sizeof(modelMat);
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
			attrDesc[index].offset = offsetof(modelMat, model) + sizeof(float) * 4 * i;
		}

		attrDesc[9].binding = 1;
		attrDesc[9].location = 9;
		attrDesc[9].format = VK_FORMAT_R32_UINT; // 1 uint32_t
		attrDesc[9].offset = offsetof(modelMat, render);

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
		vpState.pViewports = &vp;
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
		colorBA.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBA.colorBlendOp = VK_BLEND_OP_ADD;
		colorBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBA.alphaBlendOp = VK_BLEND_OP_ADD;

		std::array<VkPipelineColorBlendAttachmentState, 2> colorBAS = { colorBA, colorBA };

		VkPipelineColorBlendStateCreateInfo colorBS{};
		colorBS.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBS.logicOpEnable = VK_FALSE;
		colorBS.logicOp = VK_LOGIC_OP_COPY;
		colorBS.attachmentCount = 2;
		colorBS.pAttachments = colorBAS.data();

		// weighted color
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swap.imageFormat;
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

		// weighted alpha
		VkAttachmentDescription alphaAttachment{};
		alphaAttachment.format = swap.imageFormat;
		alphaAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		alphaAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		alphaAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		alphaAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		alphaAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		alphaAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		alphaAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkAttachmentReference alphaAttachmentRef{};
		alphaAttachmentRef.attachment = 1;
		alphaAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		std::array<VkAttachmentReference, 2> colorAttatchments = { colorAttachmentRef, alphaAttachmentRef };
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 2;
		subpass.pColorAttachments = colorAttatchments.data();

		// define the render pass
		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, alphaAttachment };
		VkRenderPassCreateInfo renderPassInf{};
		renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInf.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInf.pAttachments = attachments.data();
		renderPassInf.subpassCount = 1;
		renderPassInf.pSubpasses = &subpass;
		VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, &wboit.pipeline.renderPass);
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
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &wboit.pipeline.layout);
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
		pipelineInf.layout = wboit.pipeline.layout;
		pipelineInf.renderPass = wboit.pipeline.renderPass;
		pipelineInf.subpass = 0;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &wboit.pipeline.graphicsPipeline);
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
		vpState.pViewports = &vp; // the viewport will be the exact same as the main pass
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
		VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, &compositionPipelineData.renderPass);
		if (renderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

		VkDescriptorSetLayout setLayouts[] = { descs.layouts[5] };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &compositionPipelineData.layout);
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
		pipelineInf.layout = compositionPipelineData.layout;
		pipelineInf.renderPass = compositionPipelineData.renderPass;
		pipelineInf.subpass = 0;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &compositionPipelineData.graphicsPipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline for the composition pass!!!!!!!!");
		}
	}

	void imguiSetup() {
		// descriptor set creation for the gui:
		guiDSLayout();
		guiDSPool();

		// imgui setup:
		uint32_t graphicsQueueFamily = findQueueFamiliesG(physicalDevice).graphicsFamily.value();
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
		ImGui_ImplVulkan_Init(&initInfo, compositionPipelineData.renderPass);

		// upload fonts, etc:
		VkCommandPool guiCommandPool = createCommandPool();
		VkCommandBuffer guiCommandBuffer = beginSingleTimeCommands(guiCommandPool);
		ImGui_ImplVulkan_CreateFontsTexture(guiCommandBuffer);
		endSingleTimeCommands(guiCommandBuffer, guiCommandPool);
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
		QueueFamilyIndices queueFamilyIndices = findQueueFamiliesG(physicalDevice);
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

	void createSCCommandBuffers(std::vector<VkCommandBuffer>& cmdBuffers) {
		cmdBuffers.resize(swap.images.size());  //resize based on swap chain images size
		VkCommandBufferAllocateInfo allocInf{};
		allocInf.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInf.commandPool = commandPool; //command pool to allocate from
		allocInf.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; //primary or secondary command buffer
		allocInf.commandBufferCount = (uint32_t)cmdBuffers.size(); //number of command buffers to allocate
		if (vkAllocateCommandBuffers(device, &allocInf, cmdBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	void createSCCommandBuffers(VkCommandBuffer& cmdBuffer) {
		VkCommandBufferAllocateInfo allocInf{};
		allocInf.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInf.commandPool = commandPool; //command pool to allocate from
		allocInf.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; //primary or secondary command buffer
		allocInf.commandBufferCount = 1; //number of command buffers to allocate
		if (vkAllocateCommandBuffers(device, &allocInf, &cmdBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	uint32_t findMemoryType(uint32_t tFilter, VkMemoryPropertyFlags prop) { //find the memory type based on the type filter and properties
		VkPhysicalDeviceMemoryProperties memP; //struct to hold memory properties
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memP); //get the memory properties for the physical device
		for (uint32_t i = 0; i < memP.memoryTypeCount; i++) { //loop through the memory types
			if ((tFilter & (1 << i)) && (memP.memoryTypes[i].propertyFlags & prop) == prop) { //if the memory type is suitable
				return i; //return the index of the memory type
			}
		}
		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createModelBuffers() { // creates the vertex and index buffers for the unique models into a single buffer
		std::sort(objects.begin(), objects.end(), [](const auto& a, const auto& b) { return a->modelHash < b->modelHash; });

		bufferData.resize(getUniqueModels());
		uniqueModelIndex.clear();
		modelHashToBufferIndex.clear();

		VkDeviceSize totalVertexBufferSize = 0;
		VkDeviceSize totalIndexBufferSize = 0;

		// get the total size of the vertex and index buffers
		uint32_t ind = 0;
		for (size_t i = 0; i < objects.size(); ++i) {
			auto& obj = objects[i];
			if (uniqueModelIndex.find(obj->modelHash) == uniqueModelIndex.end()) {
				totalVertexBufferSize += sizeof(dvl::Vertex) * obj->vertices.size();
				totalIndexBufferSize += sizeof(uint32_t) * obj->indices.size();
				uniqueModelIndex[obj->modelHash] = i; //store the index of the object
				modelHashToBufferIndex[obj->modelHash] = ind++;
			}
		}

		// create and map the vertex buffer
		createBuffer(totalVertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertBuffer, vertBufferMem); // create the combined vertex buffer
		char* vertexData;
		vkMapMemory(device, vertBufferMem, 0, totalVertexBufferSize, 0, reinterpret_cast<void**>(&vertexData));
		VkDeviceSize currentVertexOffset = 0;

		// create and map the index buffer
		createBuffer(totalIndexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indBuffer, indBufferMem); // create the combined index buffer
		char* indexData;
		vkMapMemory(device, indBufferMem, 0, totalIndexBufferSize, 0, reinterpret_cast<void**>(&indexData));
		VkDeviceSize currentIndexOffset = 0;

		for (size_t i = 0; i < objects.size(); i++) {
			size_t modelInd = uniqueModelIndex[objects[i]->modelHash];
			if (modelInd != i) continue; // skip if not the first instance of the model
			size_t bufferInd = modelHashToBufferIndex[objects[i]->modelHash];

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


	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	void cloneObject(dml::vec3 pos, uint16_t object, dml::vec3 scale, dml::vec4 rotation) {
		auto m = std::make_unique<model>(*originalObjects[object]);

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
			if (m->modelHash == hash) {
				count++;
			}
		}
		return count;
	}

	size_t getUniqueModels() { // get the number of unique models
		std::unordered_set<size_t> uniqueModels;
		for (auto& m : objects) {
			uniqueModels.insert(m->modelHash);
		}
		return uniqueModels.size();
	}

	void realtimeLoad(std::string p) {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		dml::vec3 pos = dml::getCamWorldPos(cam.viewMatrix);

		cloneObject(pos, 1, { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });
		cloneObject(pos, 2, { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });

		recreateBuffers();

	}
	void recreateBuffers() {
		vkDestroyBuffer(device, vertBuffer, nullptr);
		vkFreeMemory(device, vertBufferMem, nullptr);
		vkDestroyBuffer(device, indBuffer, nullptr);
		vkFreeMemory(device, indBufferMem, nullptr);
		createModelBuffers();
	}
	void cleanupDS() {
		for (size_t i = 0; i < descs.pools.size(); i++) {
			vkDestroyDescriptorPool(device, descs.pools[i], nullptr);
		}
		for (size_t i = 0; i < descs.layouts.size(); i++) {
			vkDestroyDescriptorSetLayout(device, descs.layouts[i], nullptr);
		}
	}

	void recordCommandBuffers() { //records and submits the command buffers
		std::array<VkClearValue, 2> clearValues = { VkClearValue{0.18f, 0.3f, 0.30f, 1.0f}, VkClearValue{1.0f, 0} };
		VkDescriptorSet skyboxDescriptorSets[] = { descs.sets[3], descs.sets[4] };
		VkDescriptorSet descriptorSetsForScene[] = { descs.sets[0], descs.sets[1], descs.sets[2], descs.sets[4] };
		VkDeviceSize skyboxOffsets[] = { 0 };
		VkDeviceSize mainOffsets[] = { 0, 0 };

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;
		for (size_t i = 0; i < swap.images.size(); i++) {
			if (vkBeginCommandBuffer(mainPassCommandBuffers[i], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			vkCmdSetViewport(mainPassCommandBuffers[i], 0, 1, &vp); // set the viewport to already existing viewport state from the pipeline
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = mainPassPipeline.renderPass;
			renderPassInfo.framebuffer = mainPassFB;
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swap.extent;
			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(mainPassCommandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // begin the renderpass

			// FOR THE SKYBOX PASS
			vkCmdBindPipeline(mainPassCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipeline);
			vkCmdBindDescriptorSets(mainPassCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipelineLayout, 0, 2, skyboxDescriptorSets, 0, nullptr);
			vkCmdBindVertexBuffers(mainPassCommandBuffers[i], 0, 1, &skybox.vertBuffer, skyboxOffsets);
			vkCmdBindIndexBuffer(mainPassCommandBuffers[i], skybox.indBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(mainPassCommandBuffers[i], skybox.bufferData.indexCount, 1, skybox.bufferData.indexOffset, skybox.bufferData.vertexOffset, 0);

			// FOR THE MAIN PASS
			vkCmdBindPipeline(mainPassCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, mainPassPipeline.graphicsPipeline);
			vkCmdBindDescriptorSets(mainPassCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, mainPassPipeline.layout, 0, 4, descriptorSetsForScene, 0, nullptr);
			VkBuffer vertexBuffersArray[2] = { vertBuffer, instanceBuffer };
			VkBuffer indexBuffer = indBuffer;

			// bind the vertex and instance buffers
			vkCmdBindVertexBuffers(mainPassCommandBuffers[i], 0, 2, vertexBuffersArray, mainOffsets);
			vkCmdBindIndexBuffer(mainPassCommandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);
			uint32_t p = 0;
			for (size_t j = 0; j < objects.size(); j++) {
				uint32_t uniqueModelInd = static_cast<uint32_t>(uniqueModelIndex[objects[j]->modelHash]);
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
					vkCmdPushConstants(mainPassCommandBuffers[i], mainPassPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConst), &pushConst);

					size_t bufferInd = modelHashToBufferIndex[objects[j]->modelHash];
					uint32_t instanceCount = getModelNumHash(objects[uniqueModelInd]->modelHash);

					vkCmdDrawIndexed(mainPassCommandBuffers[i], bufferData[bufferInd].indexCount, instanceCount,
						bufferData[bufferInd].indexOffset, bufferData[bufferInd].vertexOffset, uniqueModelInd);
					p++;
				}
			}
			vkCmdEndRenderPass(mainPassCommandBuffers[i]);
			if (vkEndCommandBuffer(mainPassCommandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
			copyImage(mainPassTextures.color.image, skybox.out.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, swap.imageFormat, swap.extent.width, swap.extent.height, true);
		}
	}

	void recordCompCommandBuffers() {
		std::array<VkClearValue, 2> clearValues = { VkClearValue{0.18f, 0.3f, 0.30f, 1.0f}, VkClearValue{1.0f, 0} };
		VkDescriptorSet compDescs[] = { descs.sets[5] };

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;
		for (size_t i = 0; i < swap.images.size(); i++) {
			if (vkBeginCommandBuffer(compCommandBuffers[i], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = compositionPipelineData.renderPass;
			renderPassInfo.framebuffer = swap.framebuffers[i];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swap.extent;
			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(compCommandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(compCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, compositionPipelineData.graphicsPipeline);
			vkCmdBindDescriptorSets(compCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, compositionPipelineData.layout, 0, 1, compDescs, 0, nullptr);

			vkCmdDraw(compCommandBuffers[i], 6, 1, 0, 0);

			// prepare for next frame in ImGui:
			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			// draw the imgui text
			std::string fpsText = "fps: " + std::to_string(fps);
			std::string objText = "objects: " + std::to_string(objects.size());
			drawText(fpsText, static_cast<float>(swap.extent.width / 2), 30, font_large, ImVec4(40.0f, 61.0f, 59.0f, 0.9f));

			float w = ImGui::CalcTextSize(fpsText.c_str()).x;
			float x = static_cast<float>(swap.extent.width / 2) + w + 20.0;
			drawText(objText, x, 30, font_large, ImVec4(40.0f, 61.0f, 59.0f, 0.9f));

			// render the imgui frame and draw imgui's commands into the command buffer:
			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), compCommandBuffers[i]);

			vkCmdEndRenderPass(compCommandBuffers[i]);
			if (vkEndCommandBuffer(compCommandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	void recordWBOITCommandBuffers() {
		std::array<VkClearValue, 3> clearValues = { VkClearValue{0.0f, 0.0f, 0.0f, 1.0f}, VkClearValue{1.0f}, VkClearValue{1.0f, 0} };
		VkDescriptorSet descriptorSets[] = { descs.sets[0], descs.sets[1], descs.sets[2], descs.sets[4], descs.sets[6] };
		VkDeviceSize offset[] = { 0, 0 };

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;
		if (vkBeginCommandBuffer(wboitCommandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = wboit.pipeline.renderPass;
		renderPassInfo.framebuffer = wboit.frameBuffer;
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swap.extent;
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		transitionImageLayout(wboitCommandBuffer, mainPassTextures.depth.image, depthFormat, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, 0);

		vkCmdBeginRenderPass(wboitCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // begin the renderpass

		vkCmdBindPipeline(wboitCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wboit.pipeline.graphicsPipeline);
		vkCmdBindDescriptorSets(wboitCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wboit.pipeline.layout, 0, 5, descriptorSets, 0, nullptr);
		VkBuffer vertexBuffersArray[2] = { vertBuffer, instanceBuffer };
		VkBuffer indexBuffer = indBuffer;

		// bind the vertex and instance buffers
		vkCmdBindVertexBuffers(wboitCommandBuffer, 0, 2, vertexBuffersArray, offset);
		vkCmdBindIndexBuffer(wboitCommandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
		uint32_t p = 0;
		for (size_t j = 0; j < objects.size(); j++) {
			uint32_t uniqueModelInd = static_cast<uint32_t>(uniqueModelIndex[objects[j]->modelHash]);
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
				vkCmdPushConstants(wboitCommandBuffer, wboit.pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConst), &pushConst);

				size_t bufferInd = modelHashToBufferIndex[objects[j]->modelHash];
				uint32_t instanceCount = getModelNumHash(objects[uniqueModelInd]->modelHash);

				vkCmdDrawIndexed(wboitCommandBuffer, bufferData[bufferInd].indexCount, instanceCount,
					bufferData[bufferInd].indexOffset, bufferData[bufferInd].vertexOffset, uniqueModelInd);
				p++;
			}
		}
		vkCmdEndRenderPass(wboitCommandBuffer);
		if (vkEndCommandBuffer(wboitCommandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
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

	void createFB(VkRenderPass& renderPass, VkFramebuffer& frameBuf, VkImageView& IV, uint32_t width, uint32_t height) {
		VkFramebufferCreateInfo frameBufferInfo{};
		frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferInfo.renderPass = renderPass;
		frameBufferInfo.attachmentCount = 1;
		frameBufferInfo.pAttachments = &IV; // imageview
		frameBufferInfo.width = width;
		frameBufferInfo.height = height;
		frameBufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &frameBuf) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow framebuffer!");
		}
	}

	void createFB(VkRenderPass& renderPass, VkFramebuffer& frameBuf, std::vector<VkImageView>& attachments, uint32_t width, uint32_t height) {
		VkFramebufferCreateInfo frameBufferInfo{};
		frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferInfo.renderPass = renderPass;
		frameBufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		frameBufferInfo.pAttachments = attachments.data();
		frameBufferInfo.width = width;
		frameBufferInfo.height = height;
		frameBufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &frameBuf) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow framebuffer!");
		}
	}

	void createShadowCommandBuffers() { // create a command buffer for each light
		shadowMapCommandBuffers.resize(lights.size());
		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(shadowMapCommandBuffers.size()), shadowMapCommandBuffers.data());
		for (size_t i = 0; i < lights.size(); i++) {
			if (lights[i]->shadowMapData.frameBuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device, lights[i]->shadowMapData.frameBuffer, nullptr);
			createFB(shadowMapPipeline.renderPass, lights[i]->shadowMapData.frameBuffer, lights[i]->shadowMapData.imageView, shadowProps.mapWidth, shadowProps.mapHeight);
		}
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = static_cast<uint32_t>(lights.size());

		if (vkAllocateCommandBuffers(device, &allocInfo, shadowMapCommandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	void recordShadowCommandBuffers() {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		VkClearValue clearValue = {};
		clearValue.depthStencil.depth = 1.0f;
		clearValue.depthStencil.stencil = 0;
		for (size_t i = 0; i < lights.size(); i++) {
			if (vkBeginCommandBuffer(shadowMapCommandBuffers[i], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}
			// render pass
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = shadowMapPipeline.renderPass;
			renderPassInfo.framebuffer = lights[i]->shadowMapData.frameBuffer;
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = { shadowProps.mapWidth, shadowProps.mapHeight };
			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues = &clearValue;
			vkCmdBeginRenderPass(shadowMapCommandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(shadowMapCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipeline.graphicsPipeline);

			// bind the descriptorset that contains light matrices and the shadowmap sampler array descriptorset
			VkDescriptorSet dSets[] = { descs.sets[1] };
			vkCmdBindDescriptorSets(shadowMapCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipeline.layout, 0, 1, dSets, 0, nullptr);

			// iterate through all objects that cast shadows
			VkBuffer vertexBuffersArray[2] = { vertBuffer, instanceBuffer };
			VkBuffer indexBuffer = indBuffer;
			VkDeviceSize offsets[] = { 0, 0 };

			vkCmdBindVertexBuffers(shadowMapCommandBuffers[i], 0, 2, vertexBuffersArray, offsets);

			vkCmdBindIndexBuffer(shadowMapCommandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			for (size_t j = 0; j < objects.size(); j++) {
				uint32_t uniqueModelInd = static_cast<uint32_t>(uniqueModelIndex[objects[j]->modelHash]);
				if (uniqueModelInd == j) { // only process unique models
					struct {
						int modelIndex;
						int lightIndex;
					} pushConst;
					pushConst.modelIndex = static_cast<int>(j);
					pushConst.lightIndex = static_cast<int>(i);

					vkCmdPushConstants(shadowMapCommandBuffers[i], shadowMapPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConst), &pushConst);

					size_t bufferInd = modelHashToBufferIndex[objects[j]->modelHash];
					uint32_t instanceCount = getModelNumHash(objects[uniqueModelInd]->modelHash);
					vkCmdDrawIndexed(shadowMapCommandBuffers[i], bufferData[bufferInd].indexCount, instanceCount,
						bufferData[bufferInd].indexOffset, bufferData[bufferInd].vertexOffset, uniqueModelInd);
				}
			}

			// end the render pass and command buffer
			vkCmdEndRenderPass(shadowMapCommandBuffers[i]);
			if (vkEndCommandBuffer(shadowMapCommandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	// copy an image from one image to another
	void copyImage(VkImage& srcImage, VkImage& dstImage, VkImageLayout srcStart, VkImageLayout dstStart, VkImageLayout dstAfter, VkCommandBuffer& commandBuffer, VkFormat format, uint32_t width, uint32_t height, bool color) {
		transitionImageLayout(commandBuffer, srcImage, format, srcStart, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 1, 0);
		transitionImageLayout(commandBuffer, dstImage, format, dstStart, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 1, 0);

		VkImageCopy copy{};
		VkImageAspectFlagBits aspect = color ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
		copy.srcSubresource.aspectMask = aspect;
		copy.srcSubresource.mipLevel = 0;
		copy.srcSubresource.baseArrayLayer = 0;
		copy.srcSubresource.layerCount = 1;
		copy.dstSubresource.aspectMask = aspect;
		copy.dstSubresource.mipLevel = 0;
		copy.dstSubresource.baseArrayLayer = 0;
		copy.dstSubresource.layerCount = 1;
		copy.extent = { width, height, 1 };

		vkCmdCopyImage(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
		transitionImageLayout(commandBuffer, dstImage, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, dstAfter, 1, 1, 0);
		transitionImageLayout(commandBuffer, srcImage, format, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, 0);
	}

	void copyImage(VkImage& srcImage, VkImage& dstImage, VkImageLayout srcStart, VkImageLayout dstStart, VkImageLayout dstAfter, VkFormat format, uint32_t width, uint32_t height, bool color) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool);
		copyImage(srcImage, dstImage, srcStart, dstStart, dstAfter, commandBuffer, format, width, height, color);
		endSingleTimeCommands(commandBuffer, commandPool);
	}

	void createSkyboxBufferData() {
		createBuffer(sizeof(dml::vec3) * skybox.bufferData.vertexCount, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, skybox.vertBuffer, skybox.vertBufferMem);

		char* vertexData;
		vkMapMemory(device, skybox.vertBufferMem, 0, sizeof(dml::vec3) * skybox.bufferData.vertexCount, 0, reinterpret_cast<void**>(&vertexData));
		memcpy(vertexData, skybox.vertices.data(), sizeof(dml::vec3) * skybox.bufferData.vertexCount);
		vkUnmapMemory(device, skybox.vertBufferMem);

		createBuffer(sizeof(uint32_t) * skybox.bufferData.indexCount, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, skybox.indBuffer, skybox.indBufferMem);

		char* indexData;
		vkMapMemory(device, skybox.indBufferMem, 0, sizeof(uint32_t) * skybox.bufferData.indexCount, 0, reinterpret_cast<void**>(&indexData));
		memcpy(indexData, skybox.indices.data(), sizeof(uint32_t) * skybox.bufferData.indexCount);
		vkUnmapMemory(device, skybox.indBufferMem);
	}

	void createWBOITFB() {
		// create the framebuffer for the wboit pass
		std::vector<VkImageView> attachmentsD = { wboit.weightedColor.imageView, wboit.weightedAlpha.imageView };
		if (wboit.frameBuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device, wboit.frameBuffer, nullptr);
		createFB(wboit.pipeline.renderPass, wboit.frameBuffer, attachmentsD, swap.extent.width, swap.extent.height);
	}

	void createMainPassFramebuffers() {
		// create the framebuffers for the main pass
		std::vector<VkImageView> attachmentsD = { mainPassTextures.color.imageView, mainPassTextures.depth.imageView };
		if (mainPassFB != VK_NULL_HANDLE) vkDestroyFramebuffer(device, mainPassFB, nullptr);
		createFB(mainPassPipeline.renderPass, mainPassFB, attachmentsD, swap.extent.width, swap.extent.height);
	}

	void createFramebuffersSC() {
		// create the framebuffers for the swap chain
		swap.framebuffers.clear();
		swap.framebuffers.resize(swap.imageViews.size());
		VkImageView attachment;

		for (size_t i = 0; i < swap.imageViews.size(); ++i) {
			attachment = { swap.imageViews[i] };
			if (swap.framebuffers[i] != VK_NULL_HANDLE) vkDestroyFramebuffer(device, swap.framebuffers[i], nullptr);
			createFB(compositionPipelineData.renderPass, swap.framebuffers[i], attachment, swap.extent.width, swap.extent.height);
		}
	}

	void createSemaphores() {
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkResult resultImageAvailable = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore);
		if (resultImageAvailable != VK_SUCCESS) {
			throw std::runtime_error("failed to create image available semaphore!");
		}
		VkResult resultRenderFinished = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore);
		if (resultRenderFinished != VK_SUCCESS) {
			throw std::runtime_error("failed to create render finished semaphore!");
		}
		VkResult resultShadowFinished = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &shadowSemaphore);
		if (resultShadowFinished != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow finished semaphore!");
		}
		VkResult resultWBOITFinished = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &wboitSemaphore);
		if (resultWBOITFinished != VK_SUCCESS) {
			throw std::runtime_error("failed to create wboit finished semaphore!");
		}
		VkResult restultCompFinished = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &compSemaphore);
		if (restultCompFinished != VK_SUCCESS) {
			throw std::runtime_error("failed to create composition finished semaphore!");
		}
	}
	void freeTexture(Texture& t) {
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
		freeTexture(wboit.weightedAlpha);

		freeTexture(skybox.out);

		cleanupSwapChain();
		createSC();
		createSCImageViews();
		setupTextures();
		cleanupDS();
		setupDescriptorSets(false);

		// create the pipelines
		createGraphicsPipeline();
		createCompositionPipeline();
		createSkyboxPipeline();
		createWBOITPipeline();

		// create the framebuffers
		createMainPassFramebuffers();
		createWBOITFB();
		createFramebuffersSC();

		recordAllCommandBuffers();
		initializeMouseInput(true);
	}

	void cleanupSwapChain() { //this needs heavy modification lol
		for (auto framebuffer : swap.framebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		vkDestroyPipeline(device, mainPassPipeline.graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, mainPassPipeline.layout, nullptr);
		vkDestroyRenderPass(device, mainPassPipeline.renderPass, nullptr);
		for (auto imageView : swap.imageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swap.swapChain, nullptr);
	}


	void drawFrame() { // draw frame function
		uint32_t imageIndex;
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
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
		const size_t size = 4;
		VkSubmitInfo submitInfos[size] = {};

		// shadow pass submission
		submitInfos[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfos[0].waitSemaphoreCount = 1;
		submitInfos[0].pWaitSemaphores = &imageAvailableSemaphore;
		submitInfos[0].pWaitDstStageMask = waitStages;
		submitInfos[0].commandBufferCount = static_cast<uint32_t>(shadowMapCommandBuffers.size());
		submitInfos[0].pCommandBuffers = shadowMapCommandBuffers.data();
		submitInfos[0].signalSemaphoreCount = 1;
		submitInfos[0].pSignalSemaphores = &shadowSemaphore;

		// main scene pass submission
		submitInfos[1].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfos[1].waitSemaphoreCount = 1;
		submitInfos[1].pWaitSemaphores = &shadowSemaphore;
		submitInfos[1].pWaitDstStageMask = waitStages;
		submitInfos[1].commandBufferCount = 1;
		submitInfos[1].pCommandBuffers = &mainPassCommandBuffers[imageIndex];
		submitInfos[1].signalSemaphoreCount = 1;
		submitInfos[1].pSignalSemaphores = &wboitSemaphore;

		// wboit pass submission
		submitInfos[2].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfos[2].waitSemaphoreCount = 1;
		submitInfos[2].pWaitSemaphores = &wboitSemaphore;
		submitInfos[2].pWaitDstStageMask = waitStages;
		submitInfos[2].commandBufferCount = 1;
		submitInfos[2].pCommandBuffers = &wboitCommandBuffer;
		submitInfos[2].signalSemaphoreCount = 1;
		submitInfos[2].pSignalSemaphores = &compSemaphore;

		// composition pass submission
		submitInfos[3].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfos[3].waitSemaphoreCount = 1;
		submitInfos[3].pWaitSemaphores = &compSemaphore;
		submitInfos[3].pWaitDstStageMask = waitStages;
		submitInfos[3].commandBufferCount = 1;
		submitInfos[3].pCommandBuffers = &compCommandBuffers[imageIndex];
		submitInfos[3].signalSemaphoreCount = 1;
		submitInfos[3].pSignalSemaphores = &renderFinishedSemaphore;

		// submit both command buffers in a single call
		if (vkQueueSubmit(graphicsQueue, size, submitInfos, inFlightFences[currentFrame]) != VK_SUCCESS) {
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

	void recordAllCommandBuffers() { // record the main and shadow command buffers
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		recordShadowCommandBuffers();
		recordCommandBuffers();
		recordWBOITCommandBuffers();
		recordCompCommandBuffers();
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
			currentFrame = (currentFrame + 1) % swapSize;
			glfwPollEvents();
			drawFrame();
			handleKeyboardInput(); // handle keyboard input
			recordAllCommandBuffers();
			updateUBO(); // update ubo matrices and populate the buffer
			calcFps(startTime, previousTime, frameCount);
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
		bool isEsc = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;

		double currentFrame = glfwGetTime();
		float deltaTime = static_cast<float>(currentFrame - lastFrame);
		lastFrame = currentFrame;

		float cameraSpeed = 2.0f * deltaTime;

		cam.upAngle = fmod(cam.upAngle + 360.0f, 360.0f);
		cam.rightAngle = fmod(cam.rightAngle + 360.0f, 360.0f);

		dml::vec3 forward = cam.getLookPoint() - cam.camPos;
		dml::vec3 right = dml::cross(forward, dml::vec3(0, 1, 0));

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
			realtimeLoad("models/gear2/Gear2.obj");
		}

		if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
			uint64_t vertCount = 0;
			for (const auto& o : objects) {
				vertCount += o->vertices.size();
			}

			double startS = ((double)vertCount / fps) / 1000;
			double memEfficiency = vertCount / (1024.0 * 1024.0); // convert to mb
			double finalS = (startS) / memEfficiency;

			sep();
			std::cout << "Number of vertecies in the scene: " << vertCount << std::endl;
			std::cout << "Vertecies size: " << sizeof(dml::vec3) * vertCount << std::endl;
			std::cout << "Object count: " << objects.size() << std::endl;
			std::cout << "Score: " << finalS << std::endl;
		}

		// lock / unlock mouse
		if (isEsc && !keyPO.escPressedLastFrame) {
			cam.locked = !cam.locked;
			initializeMouseInput(cam.locked);
		}
		keyPO.escPressedLastFrame = isEsc;
	}

	void initVulkan() { //initializes Vulkan functions
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
		createModelBuffers(); //create the vertex and index buffers for the models (put them into 1)
		setupTextures();
		setupShadowMaps(); // create the inital textures for the shadow maps
		loadSkybox("overcast-skies.hdr");
		createSkyboxBufferData();
		setupModelMatInstanceBuffer();
		setupDescriptorSets(); //setup and create all the descriptor sets
		createGraphicsPipeline();
		createCompositionPipeline();
		createSkyboxPipeline();
		createShadowPipeline(); // pipeline for my shadow maps
		createWBOITPipeline();
		imguiSetup();
		updateUBO(); // populate the matrix data for the lights and objects (and put them into their designated buffer)
		createMainPassFramebuffers();
		createWBOITFB();
		createFramebuffersSC();
		createShadowCommandBuffers(); // creates the command buffers and also 1 framebuffer for each light source
		createSCCommandBuffers(mainPassCommandBuffers);
		createSCCommandBuffers(wboitCommandBuffer);
		createSCCommandBuffers(compCommandBuffers);
		recordAllCommandBuffers();
		std::cout << "Vulkan initialized successfully! Unique models: " << getUniqueModels() << std::endl;
	}

	void cleanup() { //FIX
		// destroy resources in reverse order of creation
		vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
		vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
		vkDestroySemaphore(device, shadowSemaphore, nullptr);
		for (size_t i = 0; i < 3; i++) {
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}
		for (auto frameBuffer : swap.framebuffers) {
			vkDestroyFramebuffer(device, frameBuffer, nullptr);
		}
		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(mainPassCommandBuffers.size()), mainPassCommandBuffers.data());
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		cleanupDS();
		vkDestroyPipeline(device, mainPassPipeline.graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, mainPassPipeline.layout, nullptr);
		vkDestroyRenderPass(device, mainPassPipeline.renderPass, nullptr);
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		// clean up vertex buffer and its memory
		vkDestroyBuffer(device, vertBuffer, nullptr);
		vkFreeMemory(device, vertBufferMem, nullptr);
		vkDestroyBuffer(device, indBuffer, nullptr);
		vkFreeMemory(device, indBufferMem, nullptr);
		vkDestroyBuffer(device, skybox.vertBuffer, nullptr);
		vkFreeMemory(device, skybox.vertBufferMem, nullptr);
		vkDestroyBuffer(device, skybox.indBuffer, nullptr);
		vkFreeMemory(device, skybox.indBufferMem, nullptr);

		for (auto imageView : swap.imageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swap.swapChain, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	//TODO: 
	// 1. clean up code (done)
	// 2. set up the physical and logical devices. (done)
	// 3. create a swap chain to present images to the screen (done)
	// 4. create graphics pipeline to render the triangle (done)
	// 5. commandbuffers (done)
	// 6. framebuffers (done)
	// 7. semaphores (done)
	// 8. vertex drawing and defining the vertex buffer (done)
	// 10. draw frame function (done)
	// 11. draw triangle (done)
	// 12. moving objects (done)
	// 13. fences (done)
	// 14. texture image (done)
	// 15. texture sampler (done)
	// 16. descriptor sett (done)
	// 17. convert to 3d  (done)
	// 18. mip mapping and optimizations (done)
	// 19. lighting (done)
	// 20. shadows (done)
	// 22. skybox (done)
	// 23. cleanup codebase (done)
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