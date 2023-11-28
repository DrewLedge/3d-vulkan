#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "ext/tiny_gltf.h" // load .obj and .mtl files
#include "ext/stb_image_resize.h"
#include "dml.h" // my header file with the math
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include "ext/taskflow/taskflow.hpp"

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
#define MAX_TEXTURES 4000 // temp max num of textures and models (used for passing data to shaders)
#define MAX_MODELS 1200

const uint32_t WIDTH = 3200;
const uint32_t HEIGHT = 1800;

struct camData {
	dml::vec3 camPos; //x, y, z
	dml::vec4 quat;
	float upAngle;
	float rightAngle;

	float projectionMatrix[16];
	float viewMatrix[16];

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

		return dml::rotateQ(q); // convert the quaternion to a rotation matrix
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
	struct Vertex {
		dml::vec3 pos; // position coordinates x, y, z
		dml::vec2 tex; // texture coordinates u, v
		dml::vec4 col; // color r, g, b, a
		dml::vec3 normal; // normal vector x, y, z
		float alpha;
		uint32_t matIndex; // used to know which vertex belong to which material
		dml::vec4 tangent;

		// default constructor:
		Vertex()
			: pos(dml::vec3(0.0f, 0.0f, 0.0f)),
			tex(dml::vec2(0.0f, 0.0f)),
			col(dml::vec4(0.0f, 0.0f, 0.0f, 0.0f)),
			normal(dml::vec3(0.0f, 0.0f, 0.0f)),
			alpha(1.0f),
			tangent(dml::vec4(0.0f, 0.0f, 0.0f, 0.0f))
		{}

		// constructor:
		Vertex(const dml::vec3& position,
			const dml::vec2& texture,
			const dml::vec4& color,
			const dml::vec3& normalVector,
			float alphaValue,
			const dml::vec4& tang)
			: pos(position),
			tex(texture),
			col(color),
			normal(normalVector),
			alpha(alphaValue),
			matIndex(0),
			tangent(tang)
		{}
		bool operator==(const Vertex& other) const {
			const float epsilon = 0.00001f; // tolerance for floating point equality
			return pos == other.pos &&
				tex == other.tex &&
				col == other.col &&
				normal == other.normal &&
				tangent == other.tangent &&
				std::abs(alpha - other.alpha) < epsilon;
		}
	};

	struct vertHash {
		size_t operator()(const Vertex& vertex) const {
			size_t seed = 0;

			// combine hashes for the vertex data using XOR and bit shifting:
			seed ^= std::hash<float>()(vertex.pos.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.pos.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.pos.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			seed ^= std::hash<float>()(vertex.tex.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.tex.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			seed ^= std::hash<float>()(vertex.col.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.col.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.col.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.col.w) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			seed ^= std::hash<float>()(vertex.normal.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.normal.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.normal.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			seed ^= std::hash<float>()(vertex.alpha) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			seed ^= std::hash<float>()(vertex.tangent.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.tangent.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.tangent.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.tangent.w) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			return seed;
		}
	};

	struct Texture {
		VkSampler sampler;
		VkImage image;
		VkDeviceMemory memory;
		VkImageView imageView;
		std::string path;
		uint32_t mipLevels;
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMem;
		uint32_t texIndex; //used to know what textures belong to what material
		tinygltf::Image gltfImage;
		bool found;
		uint16_t width;
		uint16_t height;

		Texture()
			: sampler(VK_NULL_HANDLE),
			image(VK_NULL_HANDLE),
			memory(VK_NULL_HANDLE),
			imageView(VK_NULL_HANDLE),
			mipLevels(1),
			stagingBuffer(VK_NULL_HANDLE),
			stagingBufferMem(VK_NULL_HANDLE),
			texIndex(0),
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
		uint32_t modelIndex; //used to know what model the material belongs to
	};
	struct model {
		std::vector<Materials> materials; //used to store all the textures/materials of the model
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		std::string pathObj; // i.e "models/cube.obj"

		dml::vec3 position;  // position of the model
		dml::vec4 rotation;  // rotation of the model in quaternions
		dml::vec3 scale;     // scale of the model
		float modelMatrix[16];

		bool isLoaded; // if object is loaded or not to prevent reloading
		bool startObj; // wether is loaded at the start of the program or not


		// default constructor:
		model()
			: materials(),
			vertices(),
			indices(),
			pathObj(""),
			position(dml::vec3(0.0f, 0.0f, 0.0f)),  // set default position to origin
			rotation(dml::vec4(0.0f, 0.0f, 0.0f, 0.0f)),  // set default rotation to no rotation
			scale(dml::vec3(0.1f, 0.1f, 0.1f)),
			isLoaded(false),
			startObj(true)
		{
			std::fill(std::begin(modelMatrix), std::end(modelMatrix), 0.0f); // initialize modelmatrix
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
		float proj[16];
		float view[16];
		float innerConeAngle; // in degrees
		float outerConeAngle; // in degrees
		float constantAttenuation;
		float linearAttenuation;
		float quadraticAttenuation;
		shadowMapDataObject shadowMapData;
	};
	struct lightMatrixUBO {
		float view[16];
		float proj[16];
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
		float model[16];
	};
	struct modelMatSSBO {
		modelMat objectMatrixData[MAX_MODELS];
	};
	struct camUBO {
		float view[16];
		float proj[16];
	};
	struct sceneIndexSSBO {
		uint32_t texIndices[MAX_TEXTURES]; //array of indices for which textures belong to what materials
		uint32_t modelIndices[MAX_MODELS]; // array of indices for which materials belong to what models

		// default constructor:
		sceneIndexSSBO() {
			// by default, all the unused indices are max + 1 so glsl can ignore them
			for (int i = 0; i < MAX_TEXTURES; i++) {
				texIndices[i] = MAX_TEXTURES + 1;
			}
			for (int i = 0; i < MAX_MODELS; i++) {
				modelIndices[i] = MAX_MODELS + 1;
			}
		}
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
	struct meshIndicies { // used so that texind, and modind are global
		uint32_t texInd = 0; // which texture belongs to which material
		uint32_t modInd = 0; // which material/textures to which mesh/model
	};
	struct sBox { // skybox struct
		Texture tex;
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
	pipelineData mainPipelineData;
	pipelineData shadowMapPipelineData;

	// command buffers and command pool
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkCommandBuffer> shadowMapCommandBuffers;
	std::vector<VkCommandBuffer> skyboxCommandBuffers;

	// buffers and related memory
	VkBuffer vertBuffer;
	VkDeviceMemory vertBufferMem;
	VkBuffer indBuffer;
	VkDeviceMemory indBufferMem;
	VkBuffer modelMatBuffer;
	VkDeviceMemory modelMatBufferMem;
	VkBuffer lightBuffer;
	VkDeviceMemory lightBufferMem;
	VkBuffer sceneIndexBuffer;
	VkDeviceMemory sceneIndexBufferMem;

	// synchronization primitives
	std::vector<VkFence> inFlightFences;
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore renderFinishedSemaphore;
	VkSemaphore shadowSemaphore;
	VkSemaphore skyboxSemaphore;

	// shader modules
	VkShaderModule fragShaderModule;
	VkShaderModule vertShaderModule;

	// depth buffering
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;
	VkFormat depthFormat;

	// descriptor sets and pools
	descriptorSetObject descs;
	VkDescriptorSetLayout imguiDescriptorSetLayout;
	VkDescriptorPool imguiDescriptorPool;

	// scene data and objects
	std::vector<bufData> bufferData;
	meshIndicies sceneInd;
	std::vector<model> objects;
	modelMatSSBO objMatData;
	camUBO camMatData;
	lightDataSSBO lightData;
	std::vector<light> lights;
	sceneIndexSSBO sceneIndices;
	shadowMapProportionsObject shadowProps;

	// textures and materials
	std::vector<Texture> allTextures;
	std::vector<Materials> allMaterials;
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

	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // enable window resizing

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForVulkan(window, true);

		font_large = ImGui::GetIO().Fonts->AddFontFromFileTTF("Fonts/OpenSans/OpenSans-Italic-VariableFont_wdth,wght.ttf", 50.0f);

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
		std::cout << " ----------------" << std::endl;
		std::cout << "model: " << stru.pathObj << std::endl;
		std::cout << "vertices: " << stru.vertices.size() << std::endl;
		std::cout << "indices: " << stru.indices.size() << std::endl;
		std::cout << "texture: " << stru.materials.size() << std::endl;
		std::cout << " ----------------" << std::endl;
	}
	void createObject(std::string path, dml::vec3 scale, dml::vec4 rotation, dml::vec3 pos) {
		loadScene(scale, pos, rotation, path);
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
		l.innerConeAngle = 30.0f;
		l.outerConeAngle = 45.0f;
		lights.push_back(l);
	}

	void loadUniqueObjects() { // load all unqiue objects and all lights
		//createObject("models/sniper_rifle_pbr.glb", { 0.3f, 0.3f, 0.3f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
		//createObject("models/sword.glb", { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
		createObject("models/knight.glb", { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f });
		createObject("models/knight.glb", { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.23f, 0.0f, 2.11f });
		//createObject("models/sniper_rifle_pbr.glb", { 0.6f, 0.6f, 0.6f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.5f });
		//createObject("models/chess.glb", { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
		createLight({ -2.0f, 0.0f, -4.0f }, { 1.0f, 1.0f, 1.0f }, 1.0f, { 0.0f, 0.0f, 0.0f });
		createLight({ -2.0f, 0.0f, 2.0f }, { 1.0f, 1.0f, 1.0f }, 1.0f, { 0.0f, 0.0f, 0.0f });
	}

	void createInstance() {
		VkApplicationInfo instanceInfo{};
		instanceInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		instanceInfo.pApplicationName = "My Engine";
		instanceInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		instanceInfo.pEngineName = "No Engine";
		instanceInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		instanceInfo.apiVersion = VK_API_VERSION_1_0;

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
	}

	void printCapabilities(VkPhysicalDeviceProperties deviceProperties) {
		std::cout << "---------------------------------" << std::endl;
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
		std::cout << "---------------------------------" << std::endl;
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
		const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
		VK_KHR_MAINTENANCE3_EXTENSION_NAME
		};

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
	std::string resultStr(VkResult result) {
		switch (result) {
		case VK_SUCCESS: return "VK_SUCCESS";
		case VK_NOT_READY: return "VK_NOT_READY";
		case VK_TIMEOUT: return "VK_TIMEOUT";
		case VK_EVENT_SET: return "VK_EVENT_SET";
		case VK_EVENT_RESET: return "VK_EVENT_RESET";
		case VK_INCOMPLETE: return "VK_INCOMPLETE";
		case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
		case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
		case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
		case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
		case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
		case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
		case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
		case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
		case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
		case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
		case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
		case VK_ERROR_FRAGMENTATION_EXT: return "VK_ERROR_FRAGMENTATION_EXT";
		case VK_ERROR_NOT_PERMITTED_EXT: return "VK_ERROR_NOT_PERMITTED_EXT";
		case VK_ERROR_INVALID_DEVICE_ADDRESS_EXT: return "VK_ERROR_INVALID_DEVICE_ADDRESS_EXT";
		case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
		case VK_THREAD_IDLE_KHR: return "VK_THREAD_IDLE_KHR";
		case VK_THREAD_DONE_KHR: return "VK_THREAD_DONE_KHR";
		case VK_OPERATION_DEFERRED_KHR: return "VK_OPERATION_DEFERRED_KHR";
		case VK_OPERATION_NOT_DEFERRED_KHR: return "VK_OPERATION_NOT_DEFERRED_KHR";
		case VK_PIPELINE_COMPILE_REQUIRED_EXT: return "VK_PIPELINE_COMPILE_REQUIRED_EXT";
		case VK_RESULT_MAX_ENUM: return "VK_RESULT_MAX_ENUM";

		default: return "Unknown VkResult";
		}
	}
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
		VkExtent2D actualExtent = { WIDTH, HEIGHT }; //extent=res
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
			throw std::runtime_error("failed to create swap chain! " + resultStr(vkCreateSwapchainKHR(device, &newinfo, nullptr, &swap.swapChain)));
		}
		// get the swap chain images
		vkGetSwapchainImagesKHR(device, swap.swapChain, &swap.imageCount, nullptr);
		swap.images.resize(swap.imageCount);
		vkGetSwapchainImagesKHR(device, swap.swapChain, &swap.imageCount, swap.images.data()); //gets the images in the swap chain
		swap.imageFormat = surfaceFormat.format;
		swap.extent = extent;
		createImageViews();
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
		dml::mat4 scaleMatrix = dml::scale(m.scale);

		// walk up the node hierarchy to accumulate transformations
		while (currentNodeIndex != -1) {
			const tinygltf::Node& node = gltfMod.nodes[currentNodeIndex];
			dml::mat4 localMatrix = calcNodeLM(node);

			// apply the scale matrix to the local matrix to scale the object
			localMatrix = scaleMatrix * localMatrix;

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

		// after accumulating all local transformations and scaling, apply the rotation and translation
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
		skybox.tex.path = path;
		createTexturedCubemap(skybox.tex);
		createTextureImgView(skybox.tex, false, "cube");
		createTS(skybox.tex, false, "cube");
		skybox.bufferData.vertexOffset = 0;
		skybox.bufferData.vertexCount = 8;
		skybox.bufferData.indexOffset = 0;
		skybox.bufferData.indexCount = 36;
	}

	void loadScene(dml::vec3 scale, dml::vec3 pos, dml::vec4 rot, std::string path) {
		tf::Executor executor;
		tf::Taskflow taskFlow;
		uint32_t meshInd = 0;


		tinygltf::Model gltfModel;
		tinygltf::TinyGLTF loader;
		std::string err;
		std::string warn;

		bool ret = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, path);
		std::cout << "-----------------------" << std::endl;
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

				std::unordered_map<Vertex, uint32_t, vertHash> uniqueVertices;
				std::vector<Vertex> tempVertices;
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
					const float* tangentData = nullptr;
					auto tangentIt = getAttributeIt("TANGENT", primitive.attributes);
					if (tangentIt != primitive.attributes.end()) { // check if the primitive has tangents
						const auto& tangentAccessor = gltfModel.accessors[tangentIt->second];
						tangentData = getAccessorData(gltfModel, primitive.attributes, "TANGENT");
					}
					else {
						tangentFound = false;
					}
					std::vector<dml::vec4> tangents(indexAccessor.count, dml::vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
					for (size_t i = 0; i < indexAccessor.count; i += 3) {
						uint32_t i1 = static_cast<const uint32_t*>(rawIndices)[i];
						uint32_t i2 = static_cast<const uint32_t*>(rawIndices)[i + 1];
						uint32_t i3 = static_cast<const uint32_t*>(rawIndices)[i + 2];

						dml::vec3 pos1 = { positionData[3 * i1], positionData[3 * i1 + 1], positionData[3 * i1 + 2] };
						dml::vec3 pos2 = { positionData[3 * i2], positionData[3 * i2 + 1], positionData[3 * i2 + 2] };
						dml::vec3 pos3 = { positionData[3 * i3], positionData[3 * i3 + 1], positionData[3 * i3 + 2] };

						dml::vec2 tex1 = { texCoordData[2 * i1], texCoordData[2 * i1 + 1] };
						dml::vec2 tex2 = { texCoordData[2 * i2], texCoordData[2 * i2 + 1] };
						dml::vec2 tex3 = { texCoordData[2 * i3], texCoordData[2 * i3 + 1] };

						dml::vec3 edge1 = pos2 - pos1;
						dml::vec3 edge2 = pos3 - pos1;
						dml::vec2 deltaUV1 = tex2 - tex1;
						dml::vec2 deltaUV2 = tex3 - tex1;

						float denominator = (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
						if (std::abs(denominator) < 1e-6) { // if the denominator is too small, skip this iteration to prevent a divide by zero
							continue;
						}
						float f = 1.0f / denominator;
						dml::vec3 tangent = (edge1 * deltaUV2.y - edge2 * deltaUV1.y) * f;

						tangents[i1].x += tangent.x;
						tangents[i1].y += tangent.y;
						tangents[i1].z += tangent.z;

						tangents[i2].x += tangent.x;
						tangents[i2].y += tangent.y;
						tangents[i2].z += tangent.z;

						tangents[i3].x += tangent.x;
						tangents[i3].y += tangent.y;
						tangents[i3].z += tangent.z;


					}
					for (dml::vec4& tangent : tangents) {
						// normalize tangent.xyz
						dml::vec3 normalizedTangent = tangent.xyz().normalize();
						tangent.x = normalizedTangent.x;
						tangent.y = normalizedTangent.y;
						tangent.z = normalizedTangent.z;
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

						Vertex vertex;
						vertex.pos = { positionData[3 * index], positionData[3 * index + 1], positionData[3 * index + 2] };
						vertex.tex = { texCoordData[2 * index], texCoordData[2 * index + 1] };
						vertex.normal = { normalData[3 * index], normalData[3 * index + 1], normalData[3 * index + 2] };
						if (colorFound) {
							vertex.col = { colorData[4 * index], colorData[4 * index + 1], colorData[4 * index + 2], colorData[4 * index + 3] };
						}
						else {
							vertex.col = { 1.0f, 1.0f, 1.0f, 1.0f };
						}

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
						vertex.matIndex = sceneInd.modInd;  // set the material index

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
							texture.baseColor.texIndex = sceneInd.texInd;
							texture.baseColor.path = "gltf";
							texture.baseColor.found = true;
						}
						else {
							std::cerr << "WARNING: Texture " << sceneInd.texInd << " doesn't have a base color texture" << std::endl;
						}

						// metallic-roughness Texture
						if (material.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
							auto& texInfo = material.pbrMetallicRoughness.metallicRoughnessTexture;
							auto& tex = gltfModel.textures[texInfo.index];
							texture.metallicRoughness.gltfImage = gltfModel.images[tex.source];
							texture.metallicRoughness.texIndex = sceneInd.texInd;
							texture.metallicRoughness.path = "gltf";
							texture.metallicRoughness.found = true;
						}
						else {
							std::cerr << "WARNING: Texture " << sceneInd.texInd << " doesn't have a metallic-roughness texture" << std::endl;
						}

						// normal map
						if (material.normalTexture.index >= 0) {
							auto& texInfo = material.normalTexture;
							auto& tex = gltfModel.textures[texInfo.index];
							texture.normalMap.gltfImage = gltfModel.images[tex.source];
							texture.normalMap.texIndex = sceneInd.texInd;
							texture.normalMap.path = "gltf";
							texture.normalMap.found = true;
						}
						else {
							std::cerr << "WARNING: Texture " << sceneInd.texInd << " doesn't have a normal map" << std::endl;
						}

						sceneInd.texInd += 1;
						texture.modelIndex = sceneInd.modInd;
						newObject.materials.push_back(texture);
					}
					else {
						std::cerr << "WARNING: Primitive " << primitive.material << " doesn't have a material/texture" << std::endl;
					}
				}
				newObject.vertices = tempVertices;
				newObject.indices = tempIndices;

				// set the newObject as loaded
				newObject.isLoaded = true;
				newObject.scale = scale;
				newObject.position = pos;
				newObject.rotation = rot;

				// calculate the model matrix for the mesh
				dml::mat4 meshModelMatrix = calcMeshWM(gltfModel, meshInd, parentInd, newObject);
				convertMatrix(meshModelMatrix, newObject.modelMatrix);

				// add newObject to global objects list
				modelMtx.lock();
				objects.push_back(newObject);
				modelMtx.unlock();

				sceneInd.modInd++;
				meshInd++;
			}
			std::cout << "Finished loading vertecies" << std::endl;
			}).name("load_model");

			auto loadTextureTask = taskFlow.emplace([&]() {
				size_t t = 0;
				for (auto& object : objects) {
					t++;
					for (size_t i = 0; i < object.materials.size(); i++) {
						//create the texture image for each texture (for each material)
						//also create mipmaps for each texture
						if (object.materials[i].baseColor.found) {
							createTexturedImage(object.materials[i].baseColor, true);
							createTextureImgView(object.materials[i].baseColor, true);
							createTS(object.materials[i].baseColor, true);
						}

						if (object.materials[i].normalMap.found) {
							createTexturedImage(object.materials[i].normalMap, false, "norm");
							createTextureImgView(object.materials[i].normalMap, false, "norm");
							createTS(object.materials[i].normalMap, false, "norm");
						}

						if (object.materials[i].metallicRoughness.found) {
							createTexturedImage(object.materials[i].metallicRoughness, false, "metallic");
							createTextureImgView(object.materials[i].metallicRoughness, false, "metallic");
							createTS(object.materials[i].metallicRoughness, false, "metallic");
						}
					}
				}
				}).name("load_texture");
				loadModelTask.precede(loadTextureTask);
				executor.run(taskFlow).get();

				std::cout << "-----------------------" << std::endl;
				std::cout << "Successfully loaded " << objects.size() << " meshes" << std::endl;
				taskFlow.clear();
	}

	void setupDepthResources() {
		depthFormat = findDepthFormat();
		createDepthImage(depthImage, depthImageMemory, swap.extent.width, swap.extent.height, depthFormat);
		depthImageView = createDepthView(depthImage, depthFormat);
	}

	void setupShadowMaps() { // initialize the shadow maps for each light
		for (size_t i = 0; i < lights.size(); i++) {
			createDepthImage(lights[i].shadowMapData.image, lights[i].shadowMapData.memory, shadowProps.mapWidth, shadowProps.mapHeight, depthFormat);
			lights[i].shadowMapData.imageView = createDepthView(lights[i].shadowMapData.image, depthFormat);
			createTS(lights[i].shadowMapData, false, "shadow");
		}
	}
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t layerCount) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool);

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
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = layerCount;

		// earliest stage in the pipeline that will wait on the barrier to be passed
		VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
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
		else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else {
			throw std::invalid_argument("Unsupported layout transition!");
		}
		vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier); // insert the barrier into the command buffer
		endSingleTimeCommands(commandBuffer, commandPool);
	}



	void createDepthImage(VkImage& image, VkDeviceMemory& imageMemory, uint32_t width, uint32_t height, VkFormat format) { //create the depth image and allocate memory for it
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;  //2d image
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.mipLevels = 1;
		imageInfo.format = format;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; //intiial layout doesn't matter
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT; //no multisampling
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; //only one queue family can access the image at a time

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create depth image!");
		}
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate depth image memory!");
		}
		vkBindImageMemory(device, image, imageMemory, 0);
	}
	VkImageView createDepthView(VkImage image, VkFormat format) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT; // only depth aspect of the image
		viewInfo.subresourceRange.baseMipLevel = 0; //no mip mapping
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create depth image view!");
		}

		return imageView;
	}



	void createImageViews() { //create the image views for the swap chain images
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
				throw std::runtime_error("Failed to create image views! Error code: " + resultStr(result));
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
			throw std::runtime_error("failed to create shader module! " + resultStr(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule)));
		}

		return shaderModule;
	}

	void setupModelMatUBO() { // ubo containing the model matricies for each object
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = sizeof(modelMatSSBO);
		bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // will be used as a storage buffer
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // will only be used by one queue family

		if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &modelMatBuffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create matrix SSBO!");
		}

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(device, modelMatBuffer, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (vkAllocateMemory(device, &allocateInfo, nullptr, &modelMatBufferMem) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate memory for the matrix SSBO!");
		}

		vkBindBufferMemory(device, modelMatBuffer, modelMatBufferMem, 0);

		// once memory is bound, map and fill it
		void* data;
		vkMapMemory(device, modelMatBufferMem, 0, bufferCreateInfo.size, 0, &data);
		memcpy(data, &objMatData, bufferCreateInfo.size);
		vkUnmapMemory(device, modelMatBufferMem);
	}

	void setupCamMatUBO() { // ubo containing the cameras matricies (view and projection)
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = sizeof(camUBO);
		bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // will be used as a storage buffer
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

	void printIndices(const sceneIndexSSBO& indexBuffer, bool extra = true) {
		if (extra) {
			for (model& obj : objects) {
				for (Vertex& vert : obj.vertices) {
					std::cout << "Vertex " << vert.pos.x << ", " << vert.pos.y << ", " << vert.pos.z << " goes with Material " << vert.matIndex << std::endl;
				}
			}
		}
		std::cout << "-------------------------------" << std::endl;
		for (size_t i = 0; i < MAX_TEXTURES; i++) {
			if (indexBuffer.texIndices[i] < MAX_TEXTURES) {
				std::cout << "Texture " << i << " goes with Material " << indexBuffer.texIndices[i] << std::endl;
			}
		}
		for (size_t i = 0; i < MAX_MODELS; i++) {
			if (indexBuffer.modelIndices[i] < MAX_MODELS) {
				std::cout << "Material " << i << " goes with Model/Object " << indexBuffer.modelIndices[i] << std::endl;
				std::cout << "Material" << i << " goes with Texture " << i * 3 << std::endl;
			}
		}
		std::cout << "-------------------------------" << std::endl;
	}

	void setupTexIndices(std::vector<Texture>& textures, std::vector<Materials>& materials) { // depricated but keeping it for now
		size_t materialCount = 0;
		for (size_t i = 0; i < totalTextureCount; ++i) {
			sceneIndices.texIndices[i] = textures[i].texIndex;
		}
		for (size_t h = 0; h < objects.size(); h++) {
			materialCount += objects[h].materials.size();
		}
		for (size_t g = 0; g < materialCount; g++) {
			sceneIndices.modelIndices[g] = materials[g].modelIndex;
		}
		//printIndices(sceneIndices);
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = sizeof(sceneIndexSSBO);
		bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // will be used as a storage buffer
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // will only be used by one queue family

		if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &sceneIndexBuffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create index buffer!");
		}

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(device, sceneIndexBuffer, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (vkAllocateMemory(device, &allocateInfo, nullptr, &sceneIndexBufferMem) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate memory for index buffer!");
		}

		vkBindBufferMemory(device, sceneIndexBuffer, sceneIndexBufferMem, 0);

		// once memory is bound, map and fill it
		void* data;
		vkMapMemory(device, sceneIndexBufferMem, 0, bufferCreateInfo.size, 0, &data);
		memcpy(data, &sceneIndices, bufferCreateInfo.size);
		vkUnmapMemory(device, sceneIndexBufferMem);
		//printIndices(sceneIndices, false); // debug
	}

	void printMatrix(const dml::mat4& matrix) { // prints the matrix in transposed order
		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++) {
				std::cout << std::fixed << std::setw(10) << std::setprecision(2) << matrix.m[i][j] << " ";
			}
			std::cout << std::endl; // end of row / column
		}
		std::cout << "---------------------------------" << std::endl;
	}

	void printVec3(const dml::vec3& vector) {
		std::cout << "{" << vector.x << ", " << vector.y << ", " << vector.z << "}" << std::endl;
	}
	void printVec4(const dml::vec4& vector) {
		std::cout << "{" << vector.x << ", " << vector.y << ", " << vector.z << ", " << vector.w << "}" << std::endl;
	}

	float getMatrixElement(const float* matrix, int col, int row) {
		return matrix[col * 4 + row];
	}

	void printFlatMatrix(const float* matrix) {
		for (int row = 0; row < 4; ++row) {  // uuter loop over rows
			for (int col = 0; col < 4; ++col) {  // inner loop over columns
				float element = getMatrixElement(matrix, col, row);
				std::cout << std::fixed << std::setw(10) << std::setprecision(4) << element << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "---------------------------------" << std::endl;
	}


	void calcCameraMats() {
		convertMatrix(cam.getViewMatrix(), cam.viewMatrix);
		convertMatrix(dml::projection(60.0f, swap.extent.width / static_cast<float>(swap.extent.height), 0.01f, 15.0f), cam.projectionMatrix);
	}

	void calcShadowMats(light& l) {
		// spotlight shadow mapping math code
		float aspectRatio = static_cast<float>(shadowProps.mapWidth) / static_cast<float>(shadowProps.mapHeight);
		float nearPlane = 0.01f, farPlane = 100.0f;

		dml::vec3 up = dml::vec3(0.0f, 1.0f, 0.0f);
		if (l.pos == l.target) {
			throw std::runtime_error("Light position and target are the same!");
		}

		dml::mat4 viewMatrix = dml::lookAt(l.pos, l.target, up);
		dml::mat4 projMatrix = dml::spotPerspective(l.outerConeAngle, aspectRatio, nearPlane, farPlane);
		/*	std::cout << "View matrix with paramiters of: pos: " << l.pos << " target: " << l.target << std::endl;
			printMatrix(viewMatrix);
			std::cout << "Projection matrix with paramiters of: angle: " << l.outerConeAngle << " aspect ratio: " << aspectRatio << " near plane: " << nearPlane << " far plane: " << farPlane << std::endl;
			printMatrix(projMatrix);*/

			//convertMatrix converts a forms::mat4 into a flat matrix and is stored in the second parameter
		convertMatrix(viewMatrix, l.view);
		convertMatrix(projMatrix, l.proj);
	}

	void updateUBO() {
		// calc matrixes for lights
		for (size_t i = 0; i < lights.size(); i++) {
			calcShadowMats(lights[i]);
			memcpy(lightData.lightsMatricies[i].proj, lights[i].proj, sizeof(lights[i].proj));
			memcpy(lightData.lightsMatricies[i].view, lights[i].view, sizeof(lights[i].view));
			copyLightToLightCords(lights[i], lightData.lightCords[i]);
		}
		void* lData;
		vkMapMemory(device, lightBufferMem, 0, sizeof(lightData), 0, &lData);
		memcpy(lData, &lightData, sizeof(lightData));
		vkUnmapMemory(device, lightBufferMem);

		// calc matricies for camera
		calcCameraMats();
		memcpy(camMatData.view, cam.viewMatrix, sizeof(cam.viewMatrix));
		memcpy(camMatData.proj, cam.projectionMatrix, sizeof(cam.projectionMatrix));

		void* cData;
		vkMapMemory(device, cam.bufferMem, 0, sizeof(camMatData), 0, &cData);
		memcpy(cData, &camMatData, sizeof(camMatData));
		vkUnmapMemory(device, cam.bufferMem);

		// calc matrixes for objects
		for (size_t i = 0; i < objects.size(); i++) {
			memcpy(objMatData.objectMatrixData[i].model, objects[i].modelMatrix, sizeof(objects[i].modelMatrix));
		}
		void* matrixData;
		vkMapMemory(device, modelMatBufferMem, 0, sizeof(objMatData), 0, &matrixData);
		memcpy(matrixData, &objMatData, sizeof(objMatData));
		vkUnmapMemory(device, modelMatBufferMem);
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

	void convertMatrix(const dml::mat4& source, float destination[16]) { //converts a 4x4 matrix to a flat array for vulkan
		size_t index = 0;
		for (size_t column = 0; column < 4; column++) {
			for (size_t row = 0; row < 4; row++) {
				destination[index] = source.m[row][column];
				index++;
			}
		}
	}

	dml::mat4 unflattenMatrix(const float source[16]) { //converts a flat array to a 4x4 matrix
		dml::mat4 destination;
		size_t index = 0;
		for (size_t column = 0; column < 4; column++) {
			for (size_t row = 0; row < 4; row++) {
				destination.m[row][column] = source[index];
				index++;
			}
		}
		return destination;
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
		size_t t = 0;
		for (const model& obj : objects) {
			t++;
			for (const Materials& materials : obj.materials) {
				// directly construct textures in-place
				allTextures.emplace_back(materials.baseColor);
				allTextures.emplace_back(materials.metallicRoughness);
				allTextures.emplace_back(materials.normalMap);
			}
		}
		std::cout << "Finished loading texture array" << std::endl;
		return allTextures;
	}

	std::vector<shadowMapDataObject> getAllShadowMaps() {
		std::vector<shadowMapDataObject>allMaps;
		allMaps.reserve(lights.size());
		for (const auto& light : lights) {
			allMaps.push_back(light.shadowMapData);
		}
		return allMaps;
	}
	std::vector<Materials> getAllMaterials() {
		size_t totalMaterials = 0;
		for (const auto& obj : objects) {
			totalMaterials += obj.materials.size();
		}
		allMaterials.reserve(totalMaterials);
		size_t t = 0;
		for (const auto& obj : objects) {
			for (auto& mat : obj.materials) {
				t += 1;
				allMaterials.push_back(mat);
			}
		}
		std::cout << "Finished loading material array" << std::endl;
		return allMaterials;
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
		int size = 6;
		descs.sets.resize(size);
		descs.layouts.resize(size);
		descs.pools.resize(size);
		uint32_t lightSize = static_cast<uint32_t>(lights.size());

		//initialize descriptor set layouts and pools
		descs.layouts[0] = createDSLayout(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT); // model matricies ssbo
		descs.layouts[1] = createDSLayout(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(totalTextureCount), VK_SHADER_STAGE_FRAGMENT_BIT); // array of textures
		descs.layouts[2] = createDSLayout(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT); // light data ssbo
		descs.layouts[3] = createDSLayout(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, lightSize, VK_SHADER_STAGE_FRAGMENT_BIT); // array of shadow map samplers
		descs.layouts[4] = createDSLayout(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT); // 1 sampler for the skybox
		descs.layouts[5] = createDSLayout(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT); // camera matricies ssbo

		descs.pools[0] = createDSPool(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
		descs.pools[1] = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(totalTextureCount));
		descs.pools[2] = createDSPool(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
		descs.pools[3] = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, lightSize);
		descs.pools[4] = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1); // skybox
		descs.pools[5] = createDSPool(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorSetCount = 1;

		std::vector<uint32_t> descCountArr = { 1, static_cast<uint32_t>(totalTextureCount), 1, lightSize, 1, 1 };

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

		setupModelMatUBO(); //create the model matrix UBOs for each object
		std::vector<shadowMapDataObject> shadowMaps = getAllShadowMaps(); // put all shadowmaps into 1 vector

		VkDescriptorBufferInfo modelMatBufferInfo{};
		modelMatBufferInfo.buffer = modelMatBuffer;
		modelMatBufferInfo.offset = 0;
		modelMatBufferInfo.range = sizeof(modelMatSSBO);

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
		VkDescriptorImageInfo skyboxInfo{};
		skyboxInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		skyboxInfo.imageView = skybox.tex.imageView;
		skyboxInfo.sampler = skybox.tex.sampler;

		VkDescriptorBufferInfo lightBufferInfo{};
		lightBufferInfo.buffer = lightBuffer;
		lightBufferInfo.offset = 0;
		lightBufferInfo.range = sizeof(lightDataSSBO);

		std::array<VkWriteDescriptorSet, 6> descriptorWrites{}; // vector to hold the info about the UBO and the texture sampler

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descs.sets[0];
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; //type=SSBO
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &modelMatBufferInfo;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descs.sets[1];
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[1].descriptorCount = static_cast<uint32_t>(totalTextureCount);
		descriptorWrites[1].pImageInfo = imageInfos.data();

		descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[2].dstSet = descs.sets[2];
		descriptorWrites[2].dstBinding = 2;
		descriptorWrites[2].dstArrayElement = 0;
		descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;//type=SSBO
		descriptorWrites[2].descriptorCount = 1;
		descriptorWrites[2].pBufferInfo = &lightBufferInfo;

		descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[3].dstSet = descs.sets[3];
		descriptorWrites[3].dstBinding = 3;
		descriptorWrites[3].dstArrayElement = 0;
		descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[3].descriptorCount = static_cast<uint32_t>(lights.size());
		descriptorWrites[3].pImageInfo = shadowInfos.data();

		descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[4].dstSet = descs.sets[4];
		descriptorWrites[4].dstBinding = 4;
		descriptorWrites[4].dstArrayElement = 0;
		descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[4].descriptorCount = 1;
		descriptorWrites[4].pImageInfo = &skyboxInfo;

		descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[5].dstSet = descs.sets[5];
		descriptorWrites[5].dstBinding = 5;
		descriptorWrites[5].dstArrayElement = 0;
		descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; //type=SSBO
		descriptorWrites[5].descriptorCount = 1;
		descriptorWrites[5].pBufferInfo = &camMatBufferInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}


	void setupDescriptorSets(bool initial = true) {
		descs.sets.clear();
		totalTextureCount = 0;
		for (const auto& object : objects) {
			totalTextureCount += object.materials.size() * 3;  // each material has 3 textures
		}
		if (initial) {
			getAllTextures();
			getAllMaterials();
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
		if (type == "shadow") {
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
		if (type == "shadow") {
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
		else if (type == "metallic") {
			viewInf.format = VK_FORMAT_R8G8B8A8_UNORM; // for metallic roughness
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else if (type == "cube") {
			viewInf.format = VK_FORMAT_R32G32B32A32_SFLOAT; // for cubemaps
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInf.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
			viewInf.subresourceRange.layerCount = 6;
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


		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = faceWidth;
		imageInfo.extent.height = faceHeight;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 6; // 6 for the six faces of a cubemap
		imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT; // HDR format
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		// create the Vulkan image
		if (vkCreateImage(device, &imageInfo, nullptr, &tex.image) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create texture image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, tex.image, &memRequirements);

		VkMemoryAllocateInfo allocInf{};
		allocInf.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInf.allocationSize = memRequirements.size;
		allocInf.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInf, nullptr, &tex.memory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate texture image memory!!!");
		}
		vkBindImageMemory(device, tex.image, tex.memory, 0);
		transitionImageLayout(tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6);
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

		transitionImageLayout(tex.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6);
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
			// create image:
			VkImageCreateInfo imageInf{};
			imageInf.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInf.imageType = VK_IMAGE_TYPE_2D;
			imageInf.extent.width = tex.width;
			imageInf.extent.height = tex.height;
			imageInf.extent.depth = 1;
			imageInf.mipLevels = tex.mipLevels;
			imageInf.arrayLayers = 1;

			if (type == "norm") {
				imageInf.format = VK_FORMAT_R8G8B8A8_UNORM;
			}
			if (type == "base") {
				imageInf.format = VK_FORMAT_R8G8B8A8_SRGB; //rgba for base texture
			}
			if (type == "metallic") {
				imageInf.format = VK_FORMAT_R8G8B8A8_UNORM;
			}
			imageInf.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInf.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInf.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
			// transfer destination bit (copy from buffer to image) and sampled bit (can be used for sampling operations)

			imageInf.samples = VK_SAMPLE_COUNT_1_BIT; // no multisampling
			imageInf.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			if (vkCreateImage(device, &imageInf, nullptr, &tex.image) != VK_SUCCESS) {
				throw std::runtime_error("failed to create texture image!");
			}

			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(device, tex.image, &memRequirements);

			VkMemoryAllocateInfo allocInf{};
			allocInf.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInf.allocationSize = memRequirements.size;
			allocInf.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			if (vkAllocateMemory(device, &allocInf, nullptr, &tex.memory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate texture image memory!!!");
			}
			vkBindImageMemory(device, tex.image, tex.memory, 0);

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

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; //specifies the layout to transition from
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; //specifies the layout to transition to
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = tex.image;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = tex.mipLevels;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; //specifies the operations that must be finished on the old layout before it transitions to the new layout
			vkCmdPipelineBarrier(tempBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier); //from the top of the pipeline to the transfer stage

			vkCmdCopyBufferToImage(tempBuffer, tex.stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region); //copy the data from the staging buffer to the image

			int mipWidth = tex.width;
			int mipHeight = tex.height;
			if (doMipmap) {
				for (uint32_t j = 0; j < tex.mipLevels; j++) {
					VkImageMemoryBarrier barrierToSrc{};
					barrierToSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
					barrierToSrc.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
					barrierToSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
					barrierToSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					barrierToSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					barrierToSrc.image = tex.image;
					barrierToSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // color image not a depth or stencil image
					barrierToSrc.subresourceRange.baseMipLevel = j;
					barrierToSrc.subresourceRange.levelCount = 1;
					barrierToSrc.subresourceRange.baseArrayLayer = 0;
					barrierToSrc.subresourceRange.layerCount = 1;
					barrierToSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; //specifies the operations that must be finished on the old layout before it transitions to the new layout
					barrierToSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
					vkCmdPipelineBarrier(tempBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrierToSrc); //from the top of the pipeline to the transfer stage

					//iterate through all but the last mip level to blit from the previous mip level to the next mip level:
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

					barrierToSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
					barrierToSrc.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					barrierToSrc.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
					barrierToSrc.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

					vkCmdPipelineBarrier(tempBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrierToSrc); //mip levels are read by the fragment shader

					//for the next mip level, divide the width and height by 2, unless they are already 1
					if (mipWidth > 1) mipWidth /= 2;
					if (mipHeight > 1) mipHeight /= 2;
				}
			}

			//transition the image from the transfer destination layout to the shader read only layout:
			for (uint32_t j = doMipmap ? tex.mipLevels : 0; j < tex.mipLevels; j++) {
				VkImageMemoryBarrier barrierToSrc{};
				barrierToSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
				barrierToSrc.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				barrierToSrc.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				barrierToSrc.image = tex.image;
				barrierToSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // color image not a depth or stencil image
				barrierToSrc.subresourceRange.baseMipLevel = j;
				barrierToSrc.subresourceRange.levelCount = 1;
				barrierToSrc.subresourceRange.baseArrayLayer = 0;
				barrierToSrc.subresourceRange.layerCount = 1;
				barrierToSrc.srcAccessMask = 0;
				barrierToSrc.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				//from the top of the pipeline to the transfer stage
				vkCmdPipelineBarrier(tempBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrierToSrc);
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
		std::vector<char> vertShaderCode = readFile("vertex_shader.spv"); //read the vertex shader binary
		std::vector<char> fragShaderCode = readFile("fragment_shader.spv");
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
		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding = 0;
		bindDesc.stride = sizeof(Vertex); // Number of bytes from one entry to the next
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // The rate when data is loaded

		std::array<VkVertexInputAttributeDescription, 7> attrDesc;

		attrDesc[0].binding = 0;
		attrDesc[0].location = 0;
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
		attrDesc[0].offset = offsetof(Vertex, pos);

		// color
		attrDesc[1].binding = 0;
		attrDesc[1].location = 1;
		attrDesc[1].format = VK_FORMAT_R32G32B32A32_SFLOAT; // 4 floats for color
		attrDesc[1].offset = offsetof(Vertex, col);

		// alpha (transparency)
		attrDesc[2].binding = 0;
		attrDesc[2].location = 2;
		attrDesc[2].format = VK_FORMAT_R32_SFLOAT; // 1 float for alpha
		attrDesc[2].offset = offsetof(Vertex, alpha);

		// texture coordinates
		attrDesc[3].binding = 0;
		attrDesc[3].location = 3;
		attrDesc[3].format = VK_FORMAT_R32G32_SFLOAT; // 2 floats for texture coordinates
		attrDesc[3].offset = offsetof(Vertex, tex);

		// material index
		attrDesc[4].binding = 0;
		attrDesc[4].location = 4;
		attrDesc[4].format = VK_FORMAT_R32_UINT; // 1 uint32_t for material index
		attrDesc[4].offset = offsetof(Vertex, matIndex);

		// normal
		attrDesc[5].binding = 0;
		attrDesc[5].location = 5;
		attrDesc[5].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for normal
		attrDesc[5].offset = offsetof(Vertex, normal);

		// tangents
		attrDesc[6].binding = 0;
		attrDesc[6].location = 6;
		attrDesc[6].format = VK_FORMAT_R32G32B32A32_SFLOAT; // 4 floats for tangent
		attrDesc[6].offset = offsetof(Vertex, tangent);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindDesc;
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

		// copy the viewport state to the heap so it can be used in the pipeline data struct
		VkPipelineViewportStateCreateInfo* viewportState = new VkPipelineViewportStateCreateInfo(vpState);

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

		//pipeline layout setup: Allows for uniform variables to be passed into the shader
		VkDescriptorSetLayout setLayouts[] = { descs.layouts[0], descs.layouts[1], descs.layouts[2], descs.layouts[3], descs.layouts[5] };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &mainPipelineData.layout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!! " + resultStr(result));
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
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //layout of the image after the render pass ends

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0; //index of the attachment description in the attachment descriptions array
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; //layout to use for the attachment during the subpass

		// define depth attachment:
		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = depthFormat;
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
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
		VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, &mainPipelineData.renderPass);
		if (renderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass! " + resultStr(renderPassResult));
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
		pipelineInf.layout = mainPipelineData.layout;
		pipelineInf.renderPass = mainPipelineData.renderPass;
		pipelineInf.subpass = 0;
		pipelineInf.basePipelineHandle = VK_NULL_HANDLE; // no base pipeline for now
		pipelineInf.basePipelineIndex = -1;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &mainPipelineData.graphicsPipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

	}
	void createShadowPipeline() {
		// get shader data
		auto vertShaderSPV = readFile("shadow_vert_shader.spv");
		auto fragShaderSPV = readFile("shadow_frag_shader.spv");
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

		// vertex input setup:
		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding = 0;
		bindDesc.stride = sizeof(Vertex);
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 1> attrDesc; // array of attribute descriptions

		// vertex position attribute
		attrDesc[0].binding = 0;
		attrDesc[0].location = 0;
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
		attrDesc[0].offset = offsetof(Vertex, pos);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindDesc;
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
		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadowMapPipelineData.renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow map render pass!");
		}

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.attachmentCount = 0;

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int) * 2; // 2 ints for the light index and the objects model matrix index

		VkDescriptorSetLayout setLayouts[] = { descs.layouts[0], descs.layouts[2] }; // the object's ubo data and the light data
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		pipelineLayoutInf.pushConstantRangeCount = 1; // one range of push constants
		pipelineLayoutInf.pPushConstantRanges = &pushConstantRange;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &shadowMapPipelineData.layout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!! " + resultStr(result));
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
		pipelineInfo.layout = shadowMapPipelineData.layout;
		pipelineInfo.renderPass = shadowMapPipelineData.renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowMapPipelineData.graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow map pipeline!!!!");
		}
	}

	void createSkyboxPipeline() { // same as the normal pipeline, but with a few small changes
		std::vector<char> vertShaderCode = readFile("sky_vert_shader.spv");
		std::vector<char> fragShaderCode = readFile("sky_frag_shader.spv");
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

		VkDescriptorSetLayout setLayouts[] = { descs.layouts[4], descs.layouts[5] };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &skybox.pipelineLayout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout for skybox!! " + resultStr(result));
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
		pipelineInf.renderPass = mainPipelineData.renderPass;
		pipelineInf.subpass = 0;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &skybox.pipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline for skybox!");
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
		ImGui_ImplVulkan_Init(&initInfo, mainPipelineData.renderPass);

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
			throw std::runtime_error("failed to create command pool! " + resultStr(result));
		}
		return cPool;
	}

	void createCommandBuffer() {
		commandBuffers.resize(swap.images.size());  //resize based on swap chain images size
		VkCommandBufferAllocateInfo allocInf{};
		allocInf.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInf.commandPool = commandPool; //command pool to allocate from
		allocInf.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; //primary or secondary command buffer
		allocInf.commandBufferCount = (uint32_t)commandBuffers.size(); //number of command buffers to allocate
		if (vkAllocateCommandBuffers(device, &allocInf, commandBuffers.data()) != VK_SUCCESS) {
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

	void createModelBuffers() { // creates the vertex and index buffers for the models into a single buffer
		bufferData.resize(objects.size());

		// get the total size of the vertex and index buffers
		VkDeviceSize totalVertexBufferSize = 0;
		VkDeviceSize totalIndexBufferSize = 0;
		for (const auto& obj : objects) {
			totalVertexBufferSize += sizeof(Vertex) * obj.vertices.size();
			totalIndexBufferSize += sizeof(uint32_t) * obj.indices.size();
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
			// vertex data
			bufferData[i].vertexOffset = static_cast<uint32_t>(currentVertexOffset);
			bufferData[i].vertexCount = static_cast<uint32_t>(objects[i].vertices.size());
			memcpy(vertexData, objects[i].vertices.data(), bufferData[i].vertexCount * sizeof(Vertex));
			vertexData += bufferData[i].vertexCount * sizeof(Vertex);
			currentVertexOffset += bufferData[i].vertexCount;

			// index data
			bufferData[i].indexOffset = static_cast<uint32_t>(currentIndexOffset);
			bufferData[i].indexCount = static_cast<uint32_t>(objects[i].indices.size());
			memcpy(indexData, objects[i].indices.data(), bufferData[i].indexCount * sizeof(uint32_t));
			indexData += bufferData[i].indexCount * sizeof(uint32_t);
			currentIndexOffset += bufferData[i].indexCount;
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
		model m = objects[object];
		m.scale = scale;
		m.position = pos;
		m.startObj = false;
		m.rotation = rotation;

		size_t objSize = objects.size();
		size_t verticesSize = m.vertices.size();
		size_t materialsSize = m.materials.size();
		size_t allMaterialsSize = allMaterials.size();

		//uint32_t texInd = -1;

		// get the texture indicies
		allMaterials.reserve(allMaterialsSize + materialsSize);
		allTextures.reserve(allMaterialsSize + 3 * materialsSize);
		for (auto& material : m.materials) {
			/*	if (material.metallicRoughness.texIndex > texInd) material.metallicRoughness.texIndex = texInd;
				if (material.baseColor.texIndex > texInd) material.baseColor.texIndex = texInd;
				if (material.normalMap.texIndex > texInd) material.normalMap.texIndex = texInd;*/

			allMaterials.emplace_back(material);
			allTextures.emplace_back(material.baseColor);
			allTextures.emplace_back(material.metallicRoughness);
			allTextures.emplace_back(material.normalMap);

			material.modelIndex = static_cast<uint32_t>(objSize);
		}

		for (size_t i = 0; i < verticesSize; i++) {
			m.vertices[i].matIndex = static_cast<uint32_t>(objSize);
		}

		dml::mat4 newModel = dml::translate(pos) * dml::rotateQ(rotation) * dml::scale(scale);
		dml::mat4 model = newModel * unflattenMatrix(m.modelMatrix);
		convertMatrix(model, m.modelMatrix);

		objects.push_back(std::move(m));

	}

	void realtimeLoad(std::string p) {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		dml::vec3 pos = dml::getCamWorldPos(unflattenMatrix(cam.viewMatrix));
		cloneObject(pos, 1, { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });
		cloneObject(pos, 0, { 0.4f, 0.4f, 0.4f }, { 0.0f, 0.0f, 0.0f, 1.0f });

		cleanupDS();
		setupDescriptorSets(false);
		createGraphicsPipeline();
		recreateBuffers();
	}
	void recreateBuffers() {
		vkDeviceWaitIdle(device);  // wait for all frames to finish
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
		VkDescriptorSet skyboxDescriptorSets[] = { descs.sets[4], descs.sets[5] };
		VkDescriptorSet descriptorSetsForScene[] = { descs.sets[0], descs.sets[1], descs.sets[2], descs.sets[3], descs.sets[5] };
		VkDeviceSize offsets[] = { 0 };
		for (size_t i = 0; i < swap.images.size(); i++) {
			vkWaitForFences(device, 1, &inFlightFences[i], VK_TRUE, UINT64_MAX);
			vkResetCommandBuffer(commandBuffers[i], 0);

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
			beginInfo.pInheritanceInfo = nullptr;
			if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}
			vkCmdSetViewport(commandBuffers[i], 0, 1, &vp); // set the viewport to already existing viewport state from the pipeline
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = mainPipelineData.renderPass;
			renderPassInfo.framebuffer = swap.framebuffers[i];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swap.extent;
			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // begin the renderpass

			// FOR THE SKYBOX
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipeline);
			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipelineLayout, 0, 2, skyboxDescriptorSets, 0, nullptr);
			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &skybox.vertBuffer, offsets);
			vkCmdBindIndexBuffer(commandBuffers[i], skybox.indBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(commandBuffers[i], skybox.bufferData.indexCount, 1, skybox.bufferData.indexOffset, skybox.bufferData.vertexOffset, 0);

			// FOR THE MAIN PASS
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, mainPipelineData.graphicsPipeline);
			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, mainPipelineData.layout, 0, 5, descriptorSetsForScene, 0, nullptr);
			VkBuffer vertexBuffersArray[1] = { vertBuffer };
			VkBuffer indexBuffer = indBuffer;
			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffersArray, offsets);
			vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);
			for (size_t j = 0; j < objects.size(); j++) {
				vkCmdDrawIndexed(commandBuffers[i], bufferData[j].indexCount, 1, bufferData[j].indexOffset, bufferData[j].vertexOffset, 0);
			}

			// prepare for next frame in ImGui:
			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			// draw the imgui text
			std::string str = std::to_string(fps);
			std::string text = "fps: " + str;
			drawText(text.c_str(), static_cast<float>(swap.extent.width / 2), 30, font_large, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));

			// render the imgui frame and draw imgui's commands into the command buffer:
			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffers[i]);

			vkCmdEndRenderPass(commandBuffers[i]);
			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}
	void createFB(VkFramebuffer& frameBuf, VkImageView IV, uint32_t width, uint32_t height) {
		VkFramebufferCreateInfo frameBufferInfo{};
		frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferInfo.renderPass = shadowMapPipelineData.renderPass;
		frameBufferInfo.attachmentCount = 1;
		frameBufferInfo.pAttachments = &IV; // imageview
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
			if (lights[i].shadowMapData.frameBuffer != VK_NULL_HANDLE) {
				vkDestroyFramebuffer(device, lights[i].shadowMapData.frameBuffer, nullptr);
			}
			createFB(lights[i].shadowMapData.frameBuffer, lights[i].shadowMapData.imageView, shadowProps.mapWidth, shadowProps.mapHeight);
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
		for (size_t i = 0; i < lights.size(); i++) {
			vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
			vkResetCommandBuffer(shadowMapCommandBuffers[i], 0);
			if (vkBeginCommandBuffer(shadowMapCommandBuffers[i], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}
			transitionImageLayout(lights[i].shadowMapData.image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
			// render pass
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = shadowMapPipelineData.renderPass;
			renderPassInfo.framebuffer = lights[i].shadowMapData.frameBuffer;
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = { shadowProps.mapWidth, shadowProps.mapHeight };

			VkClearValue clearValue = {};
			clearValue.depthStencil.depth = 1.0f;
			clearValue.depthStencil.stencil = 0;

			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues = &clearValue;
			vkCmdBeginRenderPass(shadowMapCommandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(shadowMapCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipelineData.graphicsPipeline);

			// bind the descriptorset that contains light matrices and the shadowmap sampler array descriptorset
			VkDescriptorSet dSets[] = { descs.sets[0], descs.sets[2] };
			vkCmdBindDescriptorSets(shadowMapCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipelineData.layout, 0, 2, dSets, 0, nullptr);

			// iterate through all objects that cast shadows
			VkBuffer vertexBuffersArray[1] = { vertBuffer };
			VkBuffer indexBuffer = indBuffer;
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(shadowMapCommandBuffers[i], 0, 1, vertexBuffersArray, offsets);
			vkCmdBindIndexBuffer(shadowMapCommandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			for (size_t j = 0; j < objects.size(); j++) {
				struct {
					int modelIndex;
					int lightIndex;
				} pushConst;
				pushConst.modelIndex = static_cast<int>(j);
				pushConst.lightIndex = static_cast<int>(i);

				vkCmdPushConstants(shadowMapCommandBuffers[i], shadowMapPipelineData.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConst), &pushConst);
				vkCmdDrawIndexed(shadowMapCommandBuffers[i], bufferData[j].indexCount, 1, bufferData[j].indexOffset, bufferData[j].vertexOffset, 0); // 3d models vert and index buffers
			}
			// end the render pass and transition the shadowmap image to shader read only optimal
			vkCmdEndRenderPass(shadowMapCommandBuffers[i]);
			transitionImageLayout(lights[i].shadowMapData.image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
			if (vkEndCommandBuffer(shadowMapCommandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
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


	void createFramebuffersSC(VkRenderPass renderPassF, std::vector<VkFramebuffer>& framebuffers, bool depth, VkImageView depthImageView = VK_NULL_HANDLE) {
		// create the framebuffers for the swap chain
		framebuffers.clear();
		framebuffers.resize(swap.imageViews.size());
		std::array<VkImageView, 2> attachmentsD; // with color and depth
		VkImageView attachment; // without depth

		for (size_t i = 0; i < swap.imageViews.size(); ++i) {

			if (depth) {
				attachmentsD = { swap.imageViews[i], depthImageView };
			}
			else {
				attachment = { swap.imageViews[i], };
			}

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPassF;
			if (depth) {
				framebufferInfo.attachmentCount = static_cast<uint32_t>(attachmentsD.size());
				framebufferInfo.pAttachments = attachmentsD.data();
			}
			else {
				framebufferInfo.attachmentCount = 1;
				framebufferInfo.pAttachments = &attachment;
			}
			framebufferInfo.width = swap.extent.width;
			framebufferInfo.height = swap.extent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
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
		VkResult resultSkyFinished = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &skyboxSemaphore);
		if (resultSkyFinished != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow finished semaphore!");
		}
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
		cleanupSwapChain();
		createSC();
		createImageViews();
		cleanupDS();
		setupDescriptorSets();
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);
		setupDepthResources();
		createGraphicsPipeline();
		createSkyboxPipeline();
		createFramebuffersSC(mainPipelineData.renderPass, swap.framebuffers, true, depthImageView);
		recordShadowCommandBuffers();
		recordCommandBuffers();
		initializeMouseInput(true);
	}
	void cleanupSwapChain() { //this needs heavy modification lol
		for (auto framebuffer : swap.framebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		vkDestroyPipeline(device, mainPipelineData.graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, mainPipelineData.layout, nullptr);
		vkDestroyRenderPass(device, mainPipelineData.renderPass, nullptr);
		for (auto imageView : swap.imageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swap.swapChain, nullptr);
	}

	void submitShadowCommandBuffers() { // submit the shadow command buffers to the queue
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = static_cast<uint32_t>(shadowMapCommandBuffers.size());
		submitInfo.pCommandBuffers = shadowMapCommandBuffers.data();
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit shadow command buffers!");
		}
		vkQueueWaitIdle(graphicsQueue);
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
			throw std::runtime_error("failed to acquire swap chain image! " + resultStr(result));
		}

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
		VkSemaphore shadowSignalSemaphores[] = { shadowSemaphore };
		VkSemaphore renderFinishedSemaphores[] = { renderFinishedSemaphore };

		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // stage to wait: color attachment output stage
		VkSubmitInfo submitInfo{};

		// shadow pass submission
		VkSubmitInfo shadowSubmitInfo{};
		shadowSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		shadowSubmitInfo.waitSemaphoreCount = 1;
		shadowSubmitInfo.pWaitSemaphores = waitSemaphores;
		shadowSubmitInfo.pWaitDstStageMask = waitStages;
		shadowSubmitInfo.commandBufferCount = static_cast<uint32_t>(shadowMapCommandBuffers.size());
		shadowSubmitInfo.pCommandBuffers = shadowMapCommandBuffers.data();
		shadowSubmitInfo.signalSemaphoreCount = 1;
		shadowSubmitInfo.pSignalSemaphores = shadowSignalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &shadowSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit shadow command buffer!");
		}


		// main scene pass submission
		VkSubmitInfo mainSubmitInfo{};
		mainSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		mainSubmitInfo.waitSemaphoreCount = 1;
		mainSubmitInfo.pWaitSemaphores = shadowSignalSemaphores;
		mainSubmitInfo.pWaitDstStageMask = waitStages;
		mainSubmitInfo.commandBufferCount = 1;
		mainSubmitInfo.pCommandBuffers = &commandBuffers[imageIndex];
		mainSubmitInfo.signalSemaphoreCount = 1;
		mainSubmitInfo.pSignalSemaphores = renderFinishedSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &mainSubmitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit main draw command buffer!");
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
			throw std::runtime_error("failed to present swap chain image! " + resultStr(result));
		}

		vkQueueWaitIdle(presentQueue); //wait for the queue to be idle before continuing
	}


	void recreateObjectBuffers() { // re-record command buffers to reference the new buffers
		recordShadowCommandBuffers();
		recordCommandBuffers();
	}

	void drawText(const char* text, float x, float y, ImFont* font = nullptr, ImVec4 backgroundColor = ImVec4(-1, -1, -1, -1)) {
		ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always); //set the position of the window
		ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Always); //set the size of the window

		if (backgroundColor.x != -1) {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, backgroundColor);
		}

		ImGui::Begin("TextWindow", nullptr,
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoScrollbar |
			ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoInputs |
			ImGuiWindowFlags_NoBringToFrontOnFocus);
		if (font != nullptr) { //if font exists, use it. otherwise, use the default font
			ImGui::PushFont(font);
		}
		if (font != nullptr) {
			ImGui::PopFont();
		}
		float font_size = ImGui::GetFontSize();
		float text_width = ImGui::CalcTextSize(text).x;
		float window_width = ImGui::GetWindowSize().x;
		float centered_start_position = (window_width - text_width) / 2.0f;

		ImGui::SetCursorPosX(centered_start_position); // center the text around the x position cords
		ImGui::TextUnformatted(text); // dont format the text

		if (backgroundColor.x != -1) {
			ImGui::PopStyleColor();  // revert background color change
		}
		ImGui::End();
	}
	void calcFps(auto& start, auto& prev, uint8_t& frameCount) {
		auto endTime = std::chrono::steady_clock::now();
		auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - start).count();
		frameCount++;
		start = endTime;

		auto timeSincePrevious = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - prev).count();
		if (timeSincePrevious >= 100) {
			fps = static_cast<uint32_t>(std::round(frameCount / (timeSincePrevious / 1000.0f)));
			frameCount = 0;
			prev = endTime;
		}
	}

	void mainLoop() {
		uint8_t frameCount = 0;
		auto startTime = std::chrono::steady_clock::now();
		auto previousTime = startTime;

		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
			currentFrame = (currentFrame + 1) % swap.images.size();
			handleKeyboardInput(); // handle keyboard input
			recreateObjectBuffers();
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
		}

		// only hide and capture cursor if cam.locked is true
		if (cam.locked) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}
		else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}

		// set the mouse callback
		glfwSetCursorPosCallback(window, mouseCallback);
	}



	static void mouseCallback(GLFWwindow* window, double xPos, double yPos) {
		static bool mouseFirst = true;
		float xp = static_cast<float>(xPos);
		float yp = static_cast<float>(yPos);

		if (mouseFirst) {
			cam.lastX = xp;
			cam.lastY = yp;
			mouseFirst = false;
		}

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

		// lock / unlock mouse
		if (isEsc && !keyPO.escPressedLastFrame) {
			cam.locked = !cam.locked;
			initializeMouseInput(false);
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
		setupDepthResources();
		setupShadowMaps(); // create the inital textures for the shadow maps
		loadSkybox("skyboxes/overcast-skies.hdr");
		createSkyboxBufferData();
		setupDescriptorSets(); //setup and create all the descriptor sets
		createGraphicsPipeline();
		createSkyboxPipeline();
		createShadowPipeline(); // pipeline for my shadow maps
		imguiSetup();
		updateUBO(); // populate the matrix data for the lights and objects (and put them into their designated buffer)
		createFramebuffersSC(mainPipelineData.renderPass, swap.framebuffers, true, depthImageView);
		createShadowCommandBuffers(); // creates the command buffers and also 1 framebuffer for each light source
		recordShadowCommandBuffers();
		createCommandBuffer();
		recordCommandBuffers(); //record and submit the command buffers
		//debugModelMatricies();
		std::cout << "Vulkan initialized successfully!" << std::endl;
	}
	void debugModelMatricies() {
		for (model& m : objects) {
			std::cout << "model matrix: " << std::endl;
			printFlatMatrix(m.modelMatrix);
			std::cout << "--------------------------" << std::endl;
			std::cout << "--------------------------" << std::endl;
		}
	}

	void scatterObjects(int count, float radius) {
		for (int i = 0; i < count; i++) {
			// generate random angles (rads)
			float theta = (float(rand()) / float(RAND_MAX)) * 2.0f * 3.14159f;
			float phi = acos(2.0f * (float(rand()) / float(RAND_MAX)) - 1.0f);

			// convert to cartesian coordinates
			float x = radius * sin(phi) * cos(theta);
			float y = radius * sin(phi) * sin(theta);
			float z = radius * cos(phi);

			cloneObject({ x, y, z }, 0, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f, 0.0f });
		}
	}

	void cleanup() { //FIX
		// destroy resources in reverse order of creation
		vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
		vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
		vkDestroySemaphore(device, shadowSemaphore, nullptr);
		vkDestroySemaphore(device, skyboxSemaphore, nullptr);
		for (size_t i = 0; i < 3; i++) {
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}
		for (auto frameBuffer : swap.framebuffers) {
			vkDestroyFramebuffer(device, frameBuffer, nullptr);
		}
		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		cleanupDS();
		vkDestroyPipeline(device, mainPipelineData.graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, mainPipelineData.layout, nullptr);
		vkDestroyRenderPass(device, mainPipelineData.renderPass, nullptr);
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
	// 23. cleanup codebase
	// 24. omnidirectional lighting using cubemaps
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