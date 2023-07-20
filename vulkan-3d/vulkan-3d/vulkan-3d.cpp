#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION 
#include "ext/stb_image.h" // library for loading images
#include "ext/tiny_obj_loader.h" // load .obj and .mtl files
#include "forms.h" // my header file with the math
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
		forms::vec3 pos; // position coordinates x, y, z
		forms::vec2 tex; // texture coordinates u, v
		forms::vec3 col; // color r, g, b
		forms::vec3 normal; // normal vector x, y, z
		float alpha;
		uint32_t matIndex; // used to know which vertex belong to which material

		// default constructor:
		Vertex()
			: pos(forms::vec3(0.0f, 0.0f, 0.0f)),
			tex(forms::vec2(0.0f, 0.0f)),
			col(forms::vec3(0.0f, 0.0f, 0.0f)),
			normal(forms::vec3(0.0f, 0.0f, 0.0f)),
			alpha(1.0f)
		{}

		// constructor:
		Vertex(const forms::vec3& position,
			const forms::vec2& texture,
			const forms::vec3& color,
			const forms::vec3& normalVector,
			float alphaValue)
			: pos(position),
			tex(texture),
			col(color),
			normal(normalVector),
			alpha(alphaValue),
			matIndex(0)
		{}
		bool operator==(const Vertex& other) const {
			const float epsilon = 0.00001f; // tolerance for floating point equality
			return pos == other.pos &&
				tex == other.tex &&
				col == other.col &&
				normal == other.normal &&
				std::abs(alpha - other.alpha) < epsilon;
		}
	};

	struct vertHash {
		size_t operator()(const Vertex& vertex) const {
			size_t seed = 0;

			// combine hashes for 'pos' using XOR and bit shifting:
			seed ^= std::hash<float>()(vertex.pos.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.pos.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.pos.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			seed ^= std::hash<float>()(vertex.tex.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.tex.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			seed ^= std::hash<float>()(vertex.col.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.col.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.col.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			seed ^= std::hash<float>()(vertex.normal.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.normal.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= std::hash<float>()(vertex.normal.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			seed ^= std::hash<float>()(vertex.alpha) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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

		Texture() : sampler(VK_NULL_HANDLE), image(VK_NULL_HANDLE), memory(VK_NULL_HANDLE), imageView(VK_NULL_HANDLE), mipLevels(1), stagingBuffer(VK_NULL_HANDLE), stagingBufferMem(VK_NULL_HANDLE), texIndex(0) {}

		bool operator==(const Texture& other) const {
			return sampler == other.sampler
				&& image == other.image
				&& memory == other.memory
				&& imageView == other.imageView
				&& path == other.path
				&& mipLevels == other.mipLevels
				&& stagingBuffer == other.stagingBuffer
				&& stagingBufferMem == other.stagingBufferMem;
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

			return seed;
		}
	};

	struct Materials {
		Texture diffuseTex;
		Texture specularTex;
		Texture normalMap;
		uint32_t modelIndex; //used to know what model the material belongs to
	};
	struct model {
		std::vector<Materials> materials; //used to store all the textures/materials of the model
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		std::string pathObj; // i.e "models/cube.obj"

		forms::vec3 position;  // position of the model
		forms::vec3 rotation;  // rotation of the model
		forms::vec3 scale;     // scale of the model
		float modelMatrix[16];
		float projectionMatrix[16];
		float viewMatrix[16];

		bool isLoaded; // if object is loaded or not to prevent reloading
		bool startObj; // wether is loaded at the start of the program or not


		// default constructor:
		model()
			: materials(),
			vertices(),
			indices(),
			pathObj(""),
			position(forms::vec3(0.0f, 0.0f, 0.0f)),  // set default position to origin
			rotation(forms::vec3(0.0f, 0.0f, 0.0f)),  // set default rotation to no rotation
			scale(forms::vec3(0.1f, 0.1f, 0.1f)),
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
		forms::vec3 pos;
		forms::vec3 col;
		forms::vec3 rot;
		float baseIntensity;
		float viewMatrix[16];
		float projectionMatrix[16];
		float innerConeAngle; // in radians
		float outerConeAngle; // in radians
		float constantAttenuation;
		float linearAttenuation;
		float quadraticAttenuation;
		shadowMapDataObject shadowMapData;
	};

	struct matrixUBO {
		float model[16];
		float view[16];
		float proj[16];
	};
	struct matrixDataSSBO {
		matrixUBO objectMatrixData[MAX_MODELS];
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
	struct camData {
		forms::vec3 camPos; //x, y, z
		forms::vec3 camAngle; //angle of the camera is facing
		forms::vec3 camRads;
	};
	struct shadowMapProportionsObject {
		uint32_t mapWidth = 1024;
		uint32_t mapHeight = 1024;
	};

	struct bufData {
		uint32_t vertexOffset;
		uint32_t vertexCount;
		uint32_t indexOffset;
		uint32_t indexCount;
	};

	std::vector<bufData> bufferData;
	camData cam = { forms::vec3(0.0f, 0.0f, 0.0f), forms::vec3(0.0f, 0.0f, 0.0f), forms::vec3(0.0f, 0.0f, 0.0f) };
	std::vector<light> lights = {};
	std::vector<model> objects = { };
	matrixDataSSBO matData = {};
	shadowMapProportionsObject shadowProps;

	VkSurfaceKHR surface;
	GLFWwindow* window;
	VkInstance instance;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	uint32_t imageCount;
	std::vector<VkFence> inFlightFences;
	size_t currentFrame = 0;
	std::vector<VkImageView> swapChainImageViews;
	VkViewport vp{};

	uint32_t texWidth = 512;
	uint32_t texHeight = 512;
	unsigned char* imageData;

	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;
	VkFormat depthFormat;
	sceneIndexSSBO sceneIndices = {};
	VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4; // 4 bytes per pixel
	VkDescriptorSetLayout imguiDescriptorSetLayout;
	VkDescriptorPool imguiDescriptorPool;

	std::vector<VkDescriptorSet> descriptorSets;
	VkDescriptorSetLayout descriptorSetLayout1;
	VkDescriptorPool descriptorPool1;
	VkDescriptorSetLayout descriptorSetLayout2;
	VkDescriptorPool descriptorPool2;
	VkDescriptorSetLayout descriptorSetLayout3;
	VkDescriptorPool descriptorPool3;
	VkDescriptorSetLayout descriptorSetLayout4;
	VkDescriptorPool descriptorPool4;
	VkDescriptorSetLayout descriptorSetLayout5;
	VkDescriptorPool descriptorPool5;


	VkBuffer matrixDataBuffer;
	VkDeviceMemory matrixDataBufferMem;

	VkBuffer lightBuffer;
	VkDeviceMemory lightBufferMem;
	VkBuffer sceneIndexBuffer;
	VkDeviceMemory sceneIndexBufferMem;

	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	VkPipelineLayout shadowMapPipelineLayout;
	VkPipeline shadowMapPipeline;

	VkShaderModule fragShaderModule;
	VkShaderModule vertShaderModule;
	VkRenderPass renderPass;
	VkRenderPass shadowMapRenderPass;
	VkCommandPool commandPool;

	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkCommandBuffer> shadowMapCommandBuffers;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore renderFinishedSemaphore;

	VkBuffer vertBuffer;
	VkDeviceMemory vertBufferMem;
	VkBuffer indBuffer;
	VkDeviceMemory indBufferMem;

	uint32_t fps;
	ImFont* font_small;
	ImFont* font_large;

	size_t totalTextureCount = 0;

	VkQueue presentQueue;
	VkQueue graphicsQueue;
	forms formula;

	// mutexes for multithreading
	std::mutex modelMtx;
	std::mutex mtx;

	std::array<float, 3> toArray(const forms::vec3& v) {
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
	void createObject(std::string path, forms::vec3 scale, forms::vec3 rotation, forms::vec3 pos) {
		model m;
		m.pathObj = path;
		m.scale = scale;
		m.rotation = rotation;
		m.position = pos;
		objects.push_back(m);
	}
	void createLight(forms::vec3 pos, forms::vec3 color, float intensity, forms::vec3 rot) {
		light l;
		l.col = color;
		l.pos = pos;
		l.baseIntensity = intensity;
		l.rot = rot;
		l.constantAttenuation = 1.0f;
		l.linearAttenuation = 0.1f;
		l.quadraticAttenuation = 0.032f;
		l.innerConeAngle = 0.9f; // 52 degrees
		l.outerConeAngle = 1.2f; // 68 degrees
		lights.push_back(l);
	}

	void loadUniqueObjects() { // load all unqiue objects and all lights
		createObject("models/gear/Gear1.obj", { 0.1f, 0.1f, 0.1f }, { 0.0f, 70.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
		createObject("models/gear2/Gear2.obj", { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
		createLight({ 0.0f, 0.0f, -20.0f }, { 1.0f, 1.0f, 1.0f }, 1.0f, { 0.0f, 0.0f, 0.0f });
	}

	void createInstance() {
		VkApplicationInfo instanceInfo{};
		instanceInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		instanceInfo.pApplicationName = "My Engine";
		instanceInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		instanceInfo.pEngineName = "No Engine";
		instanceInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		instanceInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo newInfo{};
		newInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		newInfo.pApplicationInfo = &instanceInfo;

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		newInfo.enabledExtensionCount = glfwExtensionCount;
		newInfo.ppEnabledExtensionNames = glfwExtensions;
		newInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		newInfo.ppEnabledLayerNames = validationLayers.data();

		if (vkCreateInstance(&newInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create instance!");
		}
	}

	void pickDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
		for (const auto& device : devices) {
			if (isDeviceSuitableG(device) && isDeviceSuitableP(device, surface)) {
				physicalDevice = device;
				VkPhysicalDeviceProperties deviceProperties;
				vkGetPhysicalDeviceProperties(device, &deviceProperties);
				printCapabilities(deviceProperties);
				break;
			}

		}
		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU for graphics and presentation");
		}
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
		queueInf.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; //creates a structure to hold que family info
		queueInf.queueFamilyIndex = indices.graphicsFamily.value(); // index of the queue family to create gotten from the findQueueFamilies function
		queueInf.queueCount = 1;
		queueInf.pQueuePriorities = &queuePriority;

		VkPhysicalDeviceFeatures deviceFeatures{};
		VkPhysicalDeviceDescriptorIndexingFeatures descIndexing{};
		descIndexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
		descIndexing.pNext = nullptr; // nullptr because we are not chaining any other structures to this one
		descIndexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE; // combined image samplers non uniform indexing
		descIndexing.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE; // ubo non uniform indexing
		descIndexing.runtimeDescriptorArray = VK_TRUE; // calculates descriptor arrays at runtime
		descIndexing.descriptorBindingVariableDescriptorCount = VK_TRUE; // descriptors can be bound by specifying a descriptor count
		descIndexing.descriptorBindingPartiallyBound = VK_TRUE; // descriptors can be partially bound which means that only some of the array elements need to be valid

		VkDeviceCreateInfo newInfo{};
		newInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		newInfo.pNext = &descIndexing; // add the indexing features to the pNext chain
		newInfo.pQueueCreateInfos = &queueInf;
		newInfo.queueCreateInfoCount = 1;
		newInfo.pEnabledFeatures = &deviceFeatures; //device features to enable
		// specify the device extensions to enable
		const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME,VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME };
		newInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		newInfo.ppEnabledExtensionNames = deviceExtensions.data();
		newInfo.enabledLayerCount = 0;
		newInfo.ppEnabledLayerNames = nullptr;
		if (vkCreateDevice(physicalDevice, &newInfo, nullptr, &device) != VK_SUCCESS) { // if logic device creation fails, output error
			throw std::runtime_error("failed to create logical device! " + resultStr(vkCreateDevice(physicalDevice, &newInfo, nullptr, &device)));
		}
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
		imageCount = swapChainSupport.capabilities.minImageCount + 1; //the number of images is based on the minimum number of images plus one
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}
		// create the swap chain.
		VkSwapchainCreateInfoKHR newinfo{};
		newinfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		newinfo.surface = surface;
		newinfo.minImageCount = imageCount;
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
		if (vkCreateSwapchainKHR(device, &newinfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain! " + resultStr(vkCreateSwapchainKHR(device, &newinfo, nullptr, &swapChain)));
		}
		// get the swap chain images
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data()); //gets the images in the swap chain
		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
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
	void loadModels() {
		tf::Executor executor;
		tf::Taskflow taskFlow;

		uint32_t texInd = 0;
		uint32_t modInd = 0;

		// parallel loading using taskflow:
		for (auto& object : objects) {
			if (!object.isLoaded) {
				auto loadModelTask = taskFlow.emplace([&]() {
					std::cout << "loading model: " << object.pathObj << std::endl;
					const std::string& objFilePath = object.pathObj;
					const std::string& mtlBasepath = objFilePath.substr(0, objFilePath.find_last_of('/') + 1);
					tinyobj::attrib_t attrib;
					std::vector<tinyobj::shape_t> shapes;
					std::vector<tinyobj::material_t> materials;
					std::string warn, err;
					tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFilePath.c_str(), mtlBasepath.c_str());

					if (!warn.empty()) {
						std::cout << "Warning: " << warn << std::endl;
					}
					if (!err.empty()) {
						throw std::runtime_error(err);
					}

					std::unordered_map<Vertex, uint32_t, vertHash> uniqueVertices;
					std::vector<Vertex> tempVertices;
					std::vector<uint32_t> tempIndices;

					// reserve memory for vectors:
					tempVertices.reserve(attrib.vertices.size() / 3);
					tempIndices.reserve(attrib.vertices.size() / 3);

					for (const auto& material : materials) {
						Materials texture;
						texture.diffuseTex.path = mtlBasepath + material.diffuse_texname;
						texture.specularTex.path = mtlBasepath + material.specular_texname;
						texture.normalMap.path = mtlBasepath + material.bump_texname;

						//get the texture index for each texture:
						texture.diffuseTex.texIndex = texInd;
						texture.specularTex.texIndex = texInd;
						texture.normalMap.texIndex = texInd;
						texInd += 1;

						//get he model index (which materials goto which model)
						texture.modelIndex = modInd;
						object.materials.push_back(texture);
					}

					for (const auto& shape : shapes) {
						uint32_t matIndex = modInd;
						for (const auto& index : shape.mesh.indices) {
							Vertex vertex;
							vertex.pos = {
								attrib.vertices[3 * index.vertex_index + 0],
								attrib.vertices[3 * index.vertex_index + 1],
								attrib.vertices[3 * index.vertex_index + 2]
							};
							vertex.tex = {
								attrib.texcoords[2 * index.texcoord_index + 0],
								1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
							};
							vertex.col = {
								attrib.colors[3 * index.vertex_index + 0],
								attrib.colors[3 * index.vertex_index + 1],
								attrib.colors[3 * index.vertex_index + 2]
							};
							vertex.normal = {
								attrib.normals[3 * index.normal_index + 0],
								attrib.normals[3 * index.normal_index + 1],
								attrib.normals[3 * index.normal_index + 2]
							};
							vertex.matIndex = matIndex;

							// Check if vertex is unique and add it to the map if it is:
							if (uniqueVertices.count(vertex) == 0) {
								uniqueVertices[vertex] = static_cast<uint32_t>(tempVertices.size());
								tempVertices.push_back(std::move(vertex));
							}
							tempIndices.push_back(uniqueVertices[vertex]);
						}
					}

					modInd += 1;

					modelMtx.lock();
					object.vertices.insert(object.vertices.end(), tempVertices.begin(), tempVertices.end());
					object.indices.insert(object.indices.end(), tempIndices.begin(), tempIndices.end());
					object.isLoaded = true;
					modelMtx.unlock();

					for (size_t i = 0; i < object.materials.size(); i++) {
						//create the texture image for each texture (for each material)
						//also create miplmaps for each texture
						createTextureImage(object.materials[i].specularTex, false);
						createTextureImage(object.materials[i].normalMap, false, "norm");
						createTextureImage(object.materials[i].diffuseTex, true);

						//create texture image views and samplers for each texture (for each material):wa
						createTextureImgView(object.materials[i].specularTex, false);
						createTS(object.materials[i].specularTex, false);

						createTextureImgView(object.materials[i].normalMap, false, "norm");
						createTS(object.materials[i].normalMap, false, "norm");

						createTextureImgView(object.materials[i].diffuseTex, true);
						createTS(object.materials[i].diffuseTex, true);
					}
					debugStruct(object);
					}).name("load_model");

			}
		}
		executor.run(taskFlow).get();
	}

	void setupDepthResources() {
		depthFormat = findDepthFormat();
		createDepthImage(depthImage, depthImageMemory, swapChainExtent.width, swapChainExtent.height, depthFormat);
		depthImageView = createDepthView(depthImage, depthFormat);
	}

	void setupShadowMaps() { // initialize the shadow maps for each light
		for (size_t i = 0; i < lights.size(); i++) {
			createDepthImage(lights[i].shadowMapData.image, lights[i].shadowMapData.memory, shadowProps.mapWidth, shadowProps.mapHeight, depthFormat);
			lights[i].shadowMapData.imageView = createDepthView(lights[i].shadowMapData.image, depthFormat);
			createTS(lights[i].shadowMapData, false, "shadow");
		}
	}
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) { // only used for depth images for now
		VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool);

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		barrier.subresourceRange.baseMipLevel = 0; // no mipmapping for depth images
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		// earliest stage in the pipeline that will wait on the barrier to be passed
		VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) { // if we are transitioning to a depth attachment
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) { // if we are transitioning to a shader read only layout for the shaders
			barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
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
		swapChainImageViews.resize(swapChainImages.size()); // resize swapChainImageViews to hold all the image views
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo newinfo{};
			newinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			newinfo.image = swapChainImages[i]; // assign the current swap chain image
			newinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			newinfo.format = swapChainImageFormat;
			newinfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY; // image will maintain its original component ordering
			newinfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			newinfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			newinfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			newinfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // set the aspect mask to color
			newinfo.subresourceRange.baseMipLevel = 0;
			newinfo.subresourceRange.levelCount = 1;
			newinfo.subresourceRange.baseArrayLayer = 0;
			newinfo.subresourceRange.layerCount = 1;
			VkResult result = vkCreateImageView(device, &newinfo, nullptr, &swapChainImageViews[i]);
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

	void setupMatrixUBO() {
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = sizeof(matrixDataSSBO);
		bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // will be used as a storage buffer
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // will only be used by one queue family

		if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &matrixDataBuffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create matrix SSBO!");
		}

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(device, matrixDataBuffer, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (vkAllocateMemory(device, &allocateInfo, nullptr, &matrixDataBufferMem) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate memory for the matrix SSBO!");
		}

		vkBindBufferMemory(device, matrixDataBuffer, matrixDataBufferMem, 0);

		// once memory is bound, map and fill it
		void* data;
		vkMapMemory(device, matrixDataBufferMem, 0, bufferCreateInfo.size, 0, &data);
		memcpy(data, &matData, bufferCreateInfo.size);
		vkUnmapMemory(device, matrixDataBufferMem);
	}
	void setupLights() {
		VkBufferCreateInfo bufferCreateInfo = {};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = sizeof(light) * lights.size();
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
		memcpy(data, lights.data(), bufferCreateInfo.size);
		vkUnmapMemory(device, lightBufferMem);
	}

	void printIndices(const sceneIndexSSBO& indexBuffer) {
		std::cout << "-------------------------------" << std::endl;
		for (size_t i = 0; i < MAX_TEXTURES; i++) {
			if (indexBuffer.texIndices[i] < MAX_TEXTURES) {
				std::cout << "Texture " << i << " goes with Material " << indexBuffer.texIndices[i] << std::endl;
			}
		}
		for (size_t i = 0; i < MAX_MODELS; i++) {
			if (indexBuffer.modelIndices[i] < MAX_MODELS) {
				std::cout << "Material " << i << " goes with Model/Object " << indexBuffer.modelIndices[i] << std::endl;
			}
		}
		std::cout << "-------------------------------" << std::endl;
	}

	void setupTexIndices(std::vector<Texture>& textures, std::vector<Materials>& materials) {
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
	}

	void calcObjectMats(model& o) {
		convertMatrix(forms::mat4::modelMatrix(o.position, o.rotation, o.scale), o.modelMatrix);
		convertMatrix(forms::mat4::viewMatrix(cam.camPos, cam.camAngle), o.viewMatrix);
		convertMatrix(forms::mat4::perspective(60.0f, swapChainExtent.width / static_cast<float>(swapChainExtent.height), 0.1f, 1000.0f), o.projectionMatrix);
		o.projectionMatrix[5] *= -1; //flip the y for vulkan
	}

	void calcShadowMats(light& l) {
		float nearPlane = 0.1f, farPlane = 200.0f;

		forms::vec3 dir = forms::vec3::getForward(forms::vec3::toRads(l.rot)); // get forward vector (direction) from the light's rotation
		forms::vec3 target = l.pos + dir;
		forms::vec3 up = forms::vec3(0.0f, 1.0f, 0.0f);

		forms::mat4 viewMatrix = forms::mat4::lookAt(l.pos, target, up);

		//convert matrix converts a forms::mat4 into a flat matrix and is stored in the second parameter
		convertMatrix(viewMatrix, l.viewMatrix);
		convertMatrix(forms::mat4::perspective(forms::vec3::toDeg(l.outerConeAngle), 1.0f, nearPlane, farPlane), l.projectionMatrix);
		l.projectionMatrix[5] *= -1; //flip the y for vulkan
	}

	void updateUBO() {
		// calc matrixes for lights
		for (size_t i = 0; i < lights.size(); i++) {
			calcShadowMats(lights[i]);
		}
		void* lightData;
		vkMapMemory(device, lightBufferMem, 0, sizeof(lights), 0, &lightData);
		memcpy(lightData, lights.data(), sizeof(lights));
		vkUnmapMemory(device, lightBufferMem);

		for (size_t i = 0; i < objects.size(); i++) {
			calcObjectMats(objects[i]);
			memcpy(matData.objectMatrixData[i].model, objects[i].modelMatrix, sizeof(objects[i].modelMatrix));
			memcpy(matData.objectMatrixData[i].view, objects[i].viewMatrix, sizeof(objects[i].viewMatrix));
			memcpy(matData.objectMatrixData[i].proj, objects[i].projectionMatrix, sizeof(objects[i].projectionMatrix));
		}
		void* matrixData;
		vkMapMemory(device, matrixDataBufferMem, 0, sizeof(matData), 0, &matrixData);
		memcpy(matrixData, &matData, sizeof(matData));
		vkUnmapMemory(device, matrixDataBufferMem);
	}
	void convertMatrix(const forms::mat4& source, float destination[16]) { //converts a 4x4 matrix to a flat array for vulkan
		size_t index = 0;
		for (size_t column = 0; column < 4; column++) {
			for (size_t row = 0; row < 4; row++) {
				destination[index] = source.m[row][column];
				index++;
			}
		}
	}
	forms::mat4 unflattenMatrix(const float source[16]) { //converts a flat array to a 4x4 matrix
		forms::mat4 destination;
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
		std::vector<Texture> allTextures;
		allTextures.reserve(totalTextureCount);

		for (const model& obj : objects) {
			for (const Materials& materials : obj.materials) {
				// diffuse texture
				allTextures.push_back(materials.diffuseTex);
				// specular texture
				allTextures.push_back(materials.specularTex);
				// normal map texture
				allTextures.push_back(materials.normalMap);
			}
		}
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
		std::vector<Materials> allMaterials;

		for (auto& obj : objects) {
			for (auto& mat : obj.materials) {
				allMaterials.push_back(mat);
			}
		}
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
		descriptorSets.resize(5);
		uint32_t lightSize = static_cast<uint32_t>(lights.size());

		//initialize descriptor set layouts and pools
		descriptorSetLayout1 = createDSLayout(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);
		descriptorSetLayout2 = createDSLayout(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(totalTextureCount), VK_SHADER_STAGE_FRAGMENT_BIT);
		descriptorSetLayout3 = createDSLayout(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);
		descriptorSetLayout4 = createDSLayout(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT); // light data ssbo
		descriptorSetLayout5 = createDSLayout(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, lightSize, VK_SHADER_STAGE_FRAGMENT_BIT); // array of shadow map samplers
		descriptorPool1 = createDSPool(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
		descriptorPool2 = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(totalTextureCount));
		descriptorPool3 = createDSPool(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
		descriptorPool4 = createDSPool(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
		descriptorPool5 = createDSPool(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, lightSize);

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorSetCount = 1; // 1 because we are allocating 1 descriptor set at a time

		std::array<VkDescriptorSetLayout, 5> layouts = { descriptorSetLayout1, descriptorSetLayout2, descriptorSetLayout3, descriptorSetLayout4,descriptorSetLayout5 };
		std::array<VkDescriptorPool, 5> pools = { descriptorPool1, descriptorPool2, descriptorPool3,descriptorPool4, descriptorPool5 };

		std::array<uint32_t, 5> descCountArr = { 1, static_cast<uint32_t>(totalTextureCount), 1 , 1,lightSize };

		for (uint32_t i = 0; i < descriptorSets.size(); i++) {
			VkDescriptorSetVariableDescriptorCountAllocateInfoEXT varCountInfo{};
			varCountInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT;
			varCountInfo.descriptorSetCount = 1;
			varCountInfo.pDescriptorCounts = &descCountArr[i];
			allocInfo.pNext = &varCountInfo; // variableCountInfo is added to the pNext chain

			allocInfo.descriptorPool = pools[i];
			allocInfo.pSetLayouts = &layouts[i];

			VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets[i]);
			if (result != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate descriptor sets. Error code: " + std::to_string(result));
			}
		}


		setupMatrixUBO(); //create the matrix UBOs for each object
		setupLights();
		std::vector<Texture> textures = getAllTextures(); // iterate through all objects and put all texture data into a vector
		std::vector<shadowMapDataObject> shadowMaps = getAllShadowMaps(); // iterate through all objects and put all texture data into a vector
		std::vector<Materials> mats = getAllMaterials(); // iterate through all objects and put all material data into a vector
		setupTexIndices(textures, mats);

		VkDescriptorBufferInfo matrixBufferInfo{};
		matrixBufferInfo.buffer = matrixDataBuffer;
		matrixBufferInfo.offset = 0;
		matrixBufferInfo.range = sizeof(matrixDataSSBO);

		std::vector<VkDescriptorImageInfo> imageInfos(totalTextureCount);
		for (size_t i = 0; i < totalTextureCount; i++) {
			imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfos[i].imageView = textures[i].imageView;
			imageInfos[i].sampler = textures[i].sampler;
		}
		std::vector<VkDescriptorImageInfo> shadowInfos(lights.size());
		for (size_t i = 0; i < lights.size(); i++) {
			shadowInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			shadowInfos[i].imageView = shadowMaps[i].imageView;
			shadowInfos[i].sampler = shadowMaps[i].sampler;
		}

		VkDescriptorBufferInfo indexBufferInfo{};
		indexBufferInfo.buffer = sceneIndexBuffer;
		indexBufferInfo.offset = 0;
		indexBufferInfo.range = sizeof(sceneIndexSSBO);

		VkDescriptorBufferInfo lightBufferInfo{};
		lightBufferInfo.buffer = lightBuffer;
		lightBufferInfo.offset = 0;
		lightBufferInfo.range = sizeof(light) * lights.size();

		std::array<VkWriteDescriptorSet, 5> descriptorWrites{}; // vector to hold the info about the UBO and the texture sampler

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descriptorSets[0];
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; //type=SSBO
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &matrixBufferInfo;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descriptorSets[1];
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[1].descriptorCount = static_cast<uint32_t>(totalTextureCount);
		descriptorWrites[1].pImageInfo = imageInfos.data();

		descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[2].dstSet = descriptorSets[2];
		descriptorWrites[2].dstBinding = 2;
		descriptorWrites[2].dstArrayElement = 0;
		descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;//type=SSBO
		descriptorWrites[2].descriptorCount = 1;
		descriptorWrites[2].pBufferInfo = &indexBufferInfo;

		descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[3].dstSet = descriptorSets[3];
		descriptorWrites[3].dstBinding = 3;
		descriptorWrites[3].dstArrayElement = 0;
		descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;//type=SSBO
		descriptorWrites[3].descriptorCount = 1;
		descriptorWrites[3].pBufferInfo = &lightBufferInfo;

		descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[4].dstSet = descriptorSets[4];
		descriptorWrites[4].dstBinding = 4;
		descriptorWrites[4].dstArrayElement = 0;
		descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
		descriptorWrites[4].descriptorCount = static_cast<uint32_t>(lights.size());
		descriptorWrites[4].pImageInfo = shadowInfos.data();

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}


	void setupDescriptorSets() {
		descriptorSets.clear();
		totalTextureCount = 0;
		for (const auto& object : objects) {
			totalTextureCount += object.materials.size() * 3;  // each material has 3 textures
		}
		createDS(); //create the descriptor set
	}
	void cloneObject(forms::vec3 pos, uint16_t object, forms::vec3 scale, forms::vec3 rotation) {
		model m = objects[object];

		// convert the CamPos to World Space
		forms::vec3 fac = 0.1f / scale;
		m.scale = scale;
		m.position = pos * fac;
		m.startObj = false;
		m.rotation = rotation;
		uint32_t objSize = static_cast<uint32_t>(objects.size());
		uint32_t texInd = -1;

		// get maximum indices from original model
		for (auto& material : m.materials) {
			if (material.diffuseTex.texIndex > texInd)
				material.diffuseTex.texIndex = texInd;
			if (material.specularTex.texIndex > texInd)
				material.specularTex.texIndex = texInd;
			if (material.normalMap.texIndex > texInd)
				material.normalMap.texIndex = texInd;
			material.modelIndex = objSize;
		}

		// increment indices for new model
		texInd++;
		for (size_t i = 0; i < m.vertices.size(); i++) {
			m.vertices[i].matIndex = objSize;
		}

		//create the model and reset the descriptor sets
		objects.push_back(m);
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
		samplerInf.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInf.minLod = 0.0f;
		samplerInf.maxLod = doMipmap ? static_cast<float>(tex.mipLevels) : 1;
		if (vkCreateSampler(device, &samplerInf, nullptr, &tex.sampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	template<typename T>
	void createTextureImgView(T& tex, bool doMipmap, std::string type = "tex") {
		VkImageViewCreateInfo viewInf{};
		viewInf.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInf.image = tex.image;
		viewInf.viewType = VK_IMAGE_VIEW_TYPE_2D;
		if (type == "shadow") {
			viewInf.format = VK_FORMAT_D32_SFLOAT;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		}
		else if (type != "tex") {
			viewInf.format = VK_FORMAT_R8G8B8A8_SNORM; // for normal maps (signed format)
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		else {
			viewInf.format = VK_FORMAT_R8G8B8A8_SRGB; // for textures like diffuse and specular
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		viewInf.subresourceRange.baseMipLevel = 0;
		viewInf.subresourceRange.levelCount = doMipmap ? tex.mipLevels - viewInf.subresourceRange.baseMipLevel : 1; //miplevel is influenced by the base 
		viewInf.subresourceRange.baseArrayLayer = 0;
		viewInf.subresourceRange.layerCount = 1;
		if (vkCreateImageView(device, &viewInf, nullptr, &tex.imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}
	}

	void getImageData(std::string path) {
		int texWidth, texHeight, texChannels;
		imageData = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		if (!imageData) {
			throw std::runtime_error("failed to load image!");
		}
	}
	void createStagingBuffer(Texture& tex) { // buffer to transfer data from the CPU (imageData) to the GPU sta
		VkBufferCreateInfo bufferInf{};
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
		memcpy(data, imageData, imageSize);
		vkUnmapMemory(device, tex.stagingBufferMem);
	}

	void createTextureImage(Texture& tex, bool doMipmap, std::string type = "tex") { //create the texture image for the diffuse texture materials
		if (tex.stagingBuffer == VK_NULL_HANDLE) {
			getImageData(tex.path);
			createStagingBuffer(tex);
			tex.mipLevels = doMipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1 : 1;
			// create image:
			VkImageCreateInfo imageInf{};
			imageInf.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInf.imageType = VK_IMAGE_TYPE_2D;
			imageInf.extent.width = texWidth;
			imageInf.extent.height = texHeight;
			imageInf.extent.depth = 1;
			imageInf.mipLevels = tex.mipLevels;
			imageInf.arrayLayers = 1;

			if (type != "tex") {
				imageInf.format = VK_FORMAT_R8G8B8A8_SNORM; //signed format for the normal maps (-1 to 1)
			}
			else {
				imageInf.format = VK_FORMAT_R8G8B8A8_SRGB; //rgba for specular and diffuse maps
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
			region.imageExtent = { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 }; //gets the 2d image extent

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

			int mipWidth = texWidth;
			int mipHeight = texHeight;
			if (doMipmap) { //only do mipmapping on the diffuse texture
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
	void createGraphicsPipelineOpaque() { //pipeline for ONLY opaque objects
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

		std::array<VkVertexInputAttributeDescription, 6> attrDesc;

		attrDesc[0].binding = 0;
		attrDesc[0].location = 0;
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
		attrDesc[0].offset = offsetof(Vertex, pos);

		// color
		attrDesc[1].binding = 0;
		attrDesc[1].location = 1;
		attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for color
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
		vp.width = static_cast<float>(swapChainExtent.width); // swap chain extent width for the viewport
		vp.height = static_cast<float>(swapChainExtent.height);
		vp.minDepth = 0.0f;
		vp.maxDepth = 1.0f;
		VkRect2D scissor{};
		scissor.offset = { 0, 0 }; // top-left corner offset
		scissor.extent = swapChainExtent;
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
		VkDescriptorSetLayout setLayouts[] = { descriptorSetLayout1, descriptorSetLayout2, descriptorSetLayout3, descriptorSetLayout4, descriptorSetLayout5 };
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &pipelineLayout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!! " + resultStr(result));
		}

		//render pass setup: Describes the attachments used by the pipeline and how many samples to use for each attachment
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat; //format of the color attachment
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
		VkResult RenderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, &renderPass);
		if (RenderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass! " + resultStr(RenderPassResult));
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
		pipelineInf.layout = pipelineLayout;
		pipelineInf.renderPass = renderPass;
		pipelineInf.subpass = 0;
		pipelineInf.basePipelineHandle = VK_NULL_HANDLE; // no base pipeline for now
		pipelineInf.basePipelineIndex = -1;
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &graphicsPipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

	}
	void createShadowPipeline() {
		// get shader data
		auto vertShaderSPV = readFile("shadow_vert_shader.spv");
		auto fragShaderSPV = readFile("shadow_frag_shader.spv");
		VkShaderModule vertShaderModule = createShaderModule(vertShaderSPV);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderSPV);

		VkPipelineShaderStageCreateInfo vertStage{};
		vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertStage.module = vertShaderModule;
		vertStage.pName = "main";
		VkPipelineShaderStageCreateInfo fragStage{};
		fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragStage.module = fragShaderModule;
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
		rasterizer.depthBiasEnable = VK_TRUE; // enable depth bias
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multiSamp{};
		multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multiSamp.sampleShadingEnable = VK_FALSE;
		multiSamp.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// creating the depth stencil state
		VkPipelineDepthStencilStateCreateInfo dStencil{};
		dStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		dStencil.depthTestEnable = VK_TRUE; //enable depth test
		dStencil.depthWriteEnable = VK_TRUE;
		dStencil.depthCompareOp = VK_COMPARE_OP_LESS;
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
		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadowMapRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow map render pass!");
		}

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.attachmentCount = 0;

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int);

		VkDescriptorSetLayout setLayouts[] = { descriptorSetLayout4 }; // the only descriptorset needed is the lights matrix data
		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = sizeof(setLayouts) / sizeof(VkDescriptorSetLayout);
		pipelineLayoutInf.pSetLayouts = setLayouts;
		pipelineLayoutInf.pushConstantRangeCount = 1; // one range of push constants
		pipelineLayoutInf.pPushConstantRanges = &pushConstantRange;
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &shadowMapPipelineLayout);
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
		pipelineInfo.layout = shadowMapPipelineLayout;
		pipelineInfo.renderPass = shadowMapRenderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowMapPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shadow map pipeline!!!!");
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
		initInfo.MinImageCount = imageCount;
		initInfo.ImageCount = imageCount;
		initInfo.CheckVkResultFn = check_vk_result; // function to check vulkan results
		ImGui_ImplVulkan_Init(&initInfo, renderPass);

		// upload fonts, etc:
		VkCommandPool guiCommandPool = createCommandPool();
		VkCommandBuffer guiCommandBuffer = beginSingleTimeCommands(guiCommandPool);
		ImGui_ImplVulkan_CreateFontsTexture(guiCommandBuffer);
		endSingleTimeCommands(guiCommandBuffer, guiCommandPool);
		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}

	void setupFences() {
		inFlightFences.resize(swapChainImages.size());
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
		commandBuffers.resize(swapChainImages.size());  //resize based on swap chain images size
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
		VkDeviceSize totalVertexBufferSize = 0;
		VkDeviceSize totalIndexBufferSize = 0;
		for (size_t i = 0; i < objects.size(); i++) {
			totalVertexBufferSize += sizeof(Vertex) * objects[i].vertices.size();
			totalIndexBufferSize += sizeof(uint32_t) * objects[i].indices.size();
		}

		createBuffer(totalVertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertBuffer, vertBufferMem); // create the combined vertex buffer

		char* vertexData;
		vkMapMemory(device, vertBufferMem, 0, totalVertexBufferSize, 0, reinterpret_cast<void**>(&vertexData)); // fill vertex buffer and save object data
		VkDeviceSize currentVertexOffset = 0;
		for (size_t i = 0; i < objects.size(); i++) {
			bufferData[i].vertexOffset = static_cast<uint32_t>(currentVertexOffset);
			bufferData[i].vertexCount = static_cast<uint32_t>(objects[i].vertices.size());
			memcpy(vertexData, objects[i].vertices.data(), bufferData[i].vertexCount * sizeof(Vertex));
			vertexData += bufferData[i].vertexCount * sizeof(Vertex);
			currentVertexOffset += bufferData[i].vertexCount;
		}

		vkUnmapMemory(device, vertBufferMem);
		createBuffer(totalIndexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indBuffer, indBufferMem); // create the combined index buffer

		// map index buffer memory into the applications address space and save the pointer in indexData as a void*
		char* indexData;
		vkMapMemory(device, indBufferMem, 0, totalIndexBufferSize, 0, reinterpret_cast<void**>(&indexData));
		VkDeviceSize currentIndexOffset = 0;

		// iterate through all objects and copy their index data into the index buffer
		for (size_t i = 0; i < objects.size(); i++) {
			bufferData[i].indexOffset = static_cast<uint32_t>(currentIndexOffset);
			bufferData[i].indexCount = static_cast<uint32_t>(objects[i].indices.size());
			memcpy(indexData, objects[i].indices.data(), bufferData[i].indexCount * sizeof(uint32_t));
			indexData += bufferData[i].indexCount * sizeof(uint32_t);
			currentIndexOffset += bufferData[i].indexCount;
		}

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

	void realtimeLoad(std::string p) {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		model m = objects[1];
		cloneObject(forms::mat4::inverseMatrix(unflattenMatrix(m.modelMatrix)).vecMatrix(cam.camPos), 1, { 0.01f, 0.01f, 0.01f }, { 0.0f, 0.0f, 0.0f });
		cleanupDS();
		setupShadowMaps();
		setupDescriptorSets();
		createGraphicsPipelineOpaque();
		recreateBuffers();
		std::cout << "Current Object Count: " << objects.size() << std::endl;
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
		vkDestroyDescriptorPool(device, descriptorPool1, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool2, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool3, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool4, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool5, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayout1, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout2, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout3, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout4, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout5, nullptr);
	}

	void recordCommandBuffers() { //records and submits the command buffers
		std::array<VkClearValue, 2> clearValues = {
		VkClearValue{0.68f, 0.85f, 0.90f, 1.0f},  // clear color: light blue
		VkClearValue{1.0f, 0}  // clear depth: 1.0f, 0 stencil
		};
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			vkWaitForFences(device, 1, &inFlightFences[i], VK_TRUE, UINT64_MAX);
			vkResetCommandBuffer(commandBuffers[i], 0);

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
			beginInfo.pInheritanceInfo = nullptr;
			if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}
			vkCmdSetViewport(commandBuffers[i], 0, 1, &vp); // Set the viewport to already existing viewport state from the pipeline
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[i];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swapChainExtent;
			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline); // bind the graphics pipeline to the command buffer

			VkDescriptorSet dSets[] = { descriptorSets[0], descriptorSets[1], descriptorSets[2], descriptorSets[3], descriptorSets[4] };
			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 5, dSets, 0, nullptr);

			VkBuffer vertexBuffersArray[1] = { vertBuffer };
			VkBuffer indexBuffer = indBuffer;
			VkDeviceSize offsets[] = { 0 };
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
			drawText(text.c_str(), static_cast<float>(swapChainExtent.width / 2), 30, font_large, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));

			// render the imgui frame and draw imgui's commands into the command buffer:
			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffers[i]);

			vkCmdEndRenderPass(commandBuffers[i]);
			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}
	void createShadowFramebuffer(VkFramebuffer& shadowFrameBuf, VkImageView depthIV, uint32_t width, uint32_t height) {
		VkFramebufferCreateInfo frameBufferInfo{};
		frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferInfo.renderPass = shadowMapRenderPass;
		frameBufferInfo.attachmentCount = 1;
		frameBufferInfo.pAttachments = &depthIV; // depth image view
		frameBufferInfo.width = width;
		frameBufferInfo.height = height;
		frameBufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &shadowFrameBuf) != VK_SUCCESS) {
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
			createShadowFramebuffer(lights[i].shadowMapData.frameBuffer, lights[i].shadowMapData.imageView, shadowProps.mapWidth, shadowProps.mapHeight);
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
	void recreateShadowMaps() { // will be used when moving lights are implemented (needs HEAVY optimization lol)
		for (auto& l : lights) {
			vkDestroyImage(device, l.shadowMapData.image, nullptr);
			vkDestroyImageView(device, l.shadowMapData.imageView, nullptr);
		}
		setupShadowMaps();
		createShadowCommandBuffers();
	}

	void recordShadowCommandBuffers() {
		//recreateShadowMaps();
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
			transitionImageLayout(lights[i].shadowMapData.image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
			// render pass
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = shadowMapRenderPass;
			renderPassInfo.framebuffer = lights[i].shadowMapData.frameBuffer;
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = { shadowProps.mapWidth, shadowProps.mapHeight };

			VkClearValue clearValue = {};
			clearValue.depthStencil.depth = 1.0f;
			clearValue.depthStencil.stencil = 0;

			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues = &clearValue;
			vkCmdBeginRenderPass(shadowMapCommandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			int lightIndex = static_cast<int>(i);
			vkCmdPushConstants(shadowMapCommandBuffers[i], shadowMapPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(int), &lightIndex);

			vkCmdBindPipeline(shadowMapCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipeline);

			// bind the descriptorset that contains light matrices and the shadowmap sampler array descriptorset
			VkDescriptorSet dSets[] = { descriptorSets[3] };
			vkCmdBindDescriptorSets(shadowMapCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipelineLayout, 0, 1, dSets, 0, nullptr);

			// iterate through all objects that cast shadows
			// this is the same code as in recordCommandBuffers()
			VkBuffer vertexBuffersArray[1] = { vertBuffer };
			VkBuffer indexBuffer = indBuffer;
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(shadowMapCommandBuffers[i], 0, 1, vertexBuffersArray, offsets);
			vkCmdBindIndexBuffer(shadowMapCommandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			for (size_t j = 0; j < objects.size(); j++) {
				vkCmdDrawIndexed(shadowMapCommandBuffers[i], bufferData[j].indexCount, 1, bufferData[j].indexOffset, bufferData[j].vertexOffset, 0);
			}

			vkCmdEndRenderPass(shadowMapCommandBuffers[i]);
			transitionImageLayout(lights[i].shadowMapData.image, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			if (vkEndCommandBuffer(shadowMapCommandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}



	void createFrameBuffer() {
		swapChainFramebuffers.resize(swapChainImageViews.size()); //resize the swap chain framebuffer vector
		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 2> attachments = {
				swapChainImageViews[i],
				depthImageView
			}; //array of attachments
			VkFramebufferCreateInfo framebufferInf{};
			framebufferInf.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInf.renderPass = renderPass; //render pass that this framebuffer will be compatible with
			framebufferInf.attachmentCount = static_cast<uint32_t>(attachments.size()); //attachment count now includes color and depth attachments
			framebufferInf.pAttachments = attachments.data(); //array of attachments
			framebufferInf.width = swapChainExtent.width; //width of the framebuffer
			framebufferInf.height = swapChainExtent.height; //height of the framebuffer
			framebufferInf.layers = 1; //1 means that each image only has one layer and there is no stereoscopic 3D
			VkResult result = vkCreateFramebuffer(device, &framebufferInf, nullptr, &swapChainFramebuffers[i]);
			if (result != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer! " + resultStr(result));
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
	}

	void recreateSwap() {
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
		vkDestroyDescriptorPool(device, descriptorPool1, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool2, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool3, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool4, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool5, nullptr);
		setupDescriptorSets();
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);
		setupDepthResources();
		createGraphicsPipelineOpaque();
		createFrameBuffer();
		recordShadowCommandBuffers();
		recordCommandBuffers();
	}
	void cleanupSwapChain() { //this needs heavy modification lol
		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
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


	void drawFrame() { //draw frame function
		uint32_t imageIndex;
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &inFlightFences[currentFrame]);
		VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex); //acquire an image from the swap chain
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			vkDeviceWaitIdle(device);
			recreateSwap();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image! " + resultStr(result));
		}

		// two semaphores for synchronization
		VkSemaphore shadowSemaphore;
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &shadowSemaphore) != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphore!");
		}

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore }; //semaphore to wait on before execution begins
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; //stage to wait: color attachment output stage
		VkSemaphore signalSemaphores[] = { shadowSemaphore }; // signal shadowSemaphore when shadow command buffer finishes execution
		VkSubmitInfo submitInfo{};

		// submit shadow command buffer
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = static_cast<uint32_t>(shadowMapCommandBuffers.size());
		submitInfo.pCommandBuffers = shadowMapCommandBuffers.data();
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit shadow command buffer!");
		}

		waitSemaphores[0] = shadowSemaphore;  // wait until shadow command buffer is done
		signalSemaphores[0] = renderFinishedSemaphore; // signal renderFinishedSemaphore when main command buffer finishes execution
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex]; //only thing that changes from shadow command buffer

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		// present the image
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;
		VkSwapchainKHR swapChains[] = { swapChain };
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
		vkDestroySemaphore(device, shadowSemaphore, nullptr);
	}


	void recreateObjectBuffers() {
		recordShadowCommandBuffers();
		recordCommandBuffers();  // re-record command buffers to reference the new buffers
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
			currentFrame = (currentFrame + 1) % swapChainImages.size();
			handleKeyboardInput(window); // handle keyboard input to change cam position
			recreateObjectBuffers();
			updateUBO(); // update ubo matrices and populate the buffer
			calcFps(startTime, previousTime, frameCount);
		}

		vkDeviceWaitIdle(device);
	}


	void handleKeyboardInput(GLFWwindow* window) {
		static double previousTime = glfwGetTime();
		double currentTime = glfwGetTime();
		float deltaTime = static_cast<float>(currentTime - previousTime) * 200;
		previousTime = currentTime;
		float cameraSpeed = 0.01f * deltaTime;
		float cameraRotationSpeed = 1.0f * deltaTime;

		cam.camAngle.y = fmod(cam.camAngle.y + 360.0f, 360.0f);
		if (cam.camAngle.x > 90) {
			cam.camAngle.x = 90;
		}
		if (cam.camAngle.x < -90) {
			cam.camAngle.x = -90;
		}

		cam.camRads = forms::vec3::toRads(cam.camAngle);
		forms::vec3 forward = cam.camPos.getForward(cam.camRads);
		forms::vec3 right = cam.camPos.getRight(cam.camRads);
		forward.y *= -1; //for vulkan

		// camera movement:
		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
			cam.camPos += forward * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			cam.camPos -= forward * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			cam.camPos += right * cameraSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			cam.camPos -= right * cameraSpeed;
		}

		// camera rotation
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			cam.camAngle.x += cameraRotationSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			cam.camAngle.x -= cameraRotationSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			cam.camAngle.y += cameraRotationSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			cam.camAngle.y -= cameraRotationSpeed;
		}
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
			realtimeLoad("models/gear2/Gear2.obj");
		}
		cam.camAngle.z = 0.0f;
	}

	void initVulkan() { //initializes Vulkan functions
		loadUniqueObjects();
		createInstance();
		createSurface();
		pickDevice();
		createLogicalDevice();
		initQueues(); //sets the queue family indices such as graphics and presentation
		createSC(); //create swap chain
		setupFences();
		createSemaphores();
		commandPool = createCommandPool();
		loadModels(); //load the model data from the obj file
		debugLights();
		createModelBuffers(); //create the vertex and index buffers for the models (put them into 1)
		setupDepthResources();
		setupShadowMaps(); // create the inital textures for the shadow maps
		setupDescriptorSets(); //setup and create all the descriptor sets
		createShadowPipeline(); // pipeline for my shadow maps
		createGraphicsPipelineOpaque();
		imguiSetup();
		updateUBO(); // populate the matrix data for the lights and objects (and put them into their designated buffer)
		createShadowCommandBuffers(); // creates the command buffers and also 1 framebuffer for each light source
		recordShadowCommandBuffers();
		createFrameBuffer();
		createCommandBuffer();
		recordCommandBuffers(); //record and submit the command buffers
		std::cout << "Vulkan initialized successfully!" << std::endl;
	}

	void debugLights() {
		for (auto& l : lights) {
			cloneObject(l.pos, 0, { 0.1f,0.1f,0.1f }, l.rot);
		}
	}
	void finishStartup() { // shadowmaps need to be recreated after the normal pipe is created for accuracy
		cleanupDS();
		setupDescriptorSets();
	}
	void testPerformance(size_t count) {
		for (size_t i = 0; i < count; ++i) {
			cloneObject({ 0.0f,0.0f,0.0f }, 0, { 0.1f,0.1f,0.1f }, { 0.0f, 0.0f, 0.0f });
		}
	}
	void cleanup() { //FIX
		// destroy resources in reverse order of creation
		vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
		vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
		for (size_t i = 0; i < 3; i++) {
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}
		for (auto frameBuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, frameBuffer, nullptr);
		}
		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		cleanupDS();
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		// clean up vertex buffer and its memory
		vkDestroyBuffer(device, vertBuffer, nullptr);
		vkFreeMemory(device, vertBufferMem, nullptr);
		vkDestroyBuffer(device, indBuffer, nullptr);
		vkFreeMemory(device, indBufferMem, nullptr);
		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
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
	// 20. shadows
	// 21. transfer all indexing code to push constants
	// 22. omnidirectional lighting using cubemaps
	// 23. skybox
	// 24. physcics
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