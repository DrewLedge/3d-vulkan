#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION 
#include "ext/stb_image.h" // library for loading images
#include "ext/tiny_obj_loader.h" // load .obj and .mtl files
#include "forms.h" // my header file with the math
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

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
#include <mutex>
#include <random>
#include <ctime> //random seed based on time
#include <chrono> // random seed based on time
#include <cmath>

const uint32_t WIDTH = 3200;
const uint32_t HEIGHT = 1800;
struct Vertex {
	forms::vec3 pos; // position coordinates x, y, z
	forms::vec2 tex; // texture coordinates u, v
	forms::vec3 col; // color r, g, b
	forms::vec3 normal; // normal vector x, y, z
	float alpha;

	// default constructor:
	Vertex()
		: pos(forms::vec3(0.0f, 0.0f, 0.0f)),
		tex(forms::vec2(0.0f, 0.0f)),
		col(forms::vec3(0.0f, 0.0f, 0.0f)),
		normal(forms::vec3(0.0f, 0.0f, 0.0f)),
		alpha(0.95f)
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
		alpha(alphaValue)
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
	VkSampler textureSampler;
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	uint32_t mipLevels;
	// texture file paths
	std::string ambientTexturePath;
	std::string diffuseTexturePath;
	std::string specularTexturePath;
	std::string normalMapPath;
	std::string displacementMapPath;
	std::string bumpMapPath;

	// default constructor
	Texture()
		: textureSampler(VK_NULL_HANDLE),
		textureImage(VK_NULL_HANDLE),
		textureImageMemory(VK_NULL_HANDLE),
		textureImageView(VK_NULL_HANDLE)
	{}
};
struct model {
	Texture texture;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::string pathObj; // i.e "models/cube.obj"
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMem;

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
		: texture(),
		vertices(),
		indices(),
		pathObj(""),
		position(forms::vec3(0.0f, 0.0f, 0.0f)),  // set default position to origin
		rotation(forms::vec3(0.0f, 0.0f, 0.0f)),  // set default rotation to no rotation
		scale(forms::vec3(0.1f, 0.1f, 0.1f)),
		isLoaded(false),
		stagingBuffer(VK_NULL_HANDLE),
		stagingBufferMem(VK_NULL_HANDLE),
		startObj(true)
	{
		std::fill(std::begin(modelMatrix), std::end(modelMatrix), 0.0f); // initialize modelmatrix
	}
};


model model1 = {};
model model2 = {};
std::vector<model> objects = { model1, model2 };

struct UniformBufferObject {
	float model[16];
	float view[16];
	float proj[16];
};

struct camData {
	forms::vec3 camPos; //x, y, z
	forms::vec3 camAngle; //angle of the camera is facing
	forms::vec3 camRads;
};
camData cam = { forms::vec3(0.0f, 0.0f, 0.0f), forms::vec3(0.0f, 0.0f, 0.0f), forms::vec3(0.0f, 0.0f, 0.0f) };
class Engine {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}
private:
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

	VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4; // gets height and width of image and multiplies them by 4 (4 bytes per pixel)
	VkDescriptorSetLayout descriptorSetLayout; //descriptor set layout object, defined in the pipeline
	VkDescriptorPool descriptorPool; // descriptor pool object
	VkDescriptorSetLayout imguiDescriptorSetLayout;
	VkDescriptorPool imguiDescriptorPool;

	std::vector<VkDescriptorSet> descriptorSets;
	std::vector<VkBuffer> uboBuffers;
	std::vector<VkDeviceMemory> uboBufferMemories;

	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	VkShaderModule fragShaderModule;
	VkShaderModule vertShaderModule;
	VkRenderPass renderPass;
	VkCommandPool commandPool;

	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore renderFinishedSemaphore;

	std::vector<VkBuffer> vertBuffers;
	std::vector<VkDeviceMemory> vertBufferMems;
	std::vector<VkBuffer> indBuffers;
	std::vector<VkDeviceMemory> indBufferMems;

	VkQueue presentQueue;
	VkQueue graphicsQueue;
	forms formula;
	std::mutex modelMtx;
	std::mutex descMtx;
	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // enable window resizing

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
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

	void loadModels() {
		std::vector<std::thread> threads;

		// parallel loading using multiple threads:
		for (auto& object : objects) {
			if (!object.isLoaded) {
				threads.push_back(std::thread([&]() { // lambda function to create a new thread to load the model
					std::cout << "loading model: " << object.pathObj << std::endl;
					const std::string& objFilePath = object.pathObj;
					const std::string& mtl_basepath = objFilePath.substr(0, objFilePath.find_last_of('/') + 1);
					tinyobj::attrib_t attrib;
					std::vector<tinyobj::shape_t> shapes;
					std::vector<tinyobj::material_t> materials;
					std::string warn, err;
					tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFilePath.c_str(), mtl_basepath.c_str());
					if (!warn.empty()) {
						std::cout << "Warning: " << warn << std::endl;
					}
					if (!err.empty()) {
						throw std::runtime_error(err);
					}
					// Use an unordered_map instead of a vector for storing unique vertices:
					std::unordered_map<Vertex, uint32_t, vertHash> uniqueVertices;

					std::vector<Vertex> tempVertices;
					std::vector<uint32_t> tempIndices;

					// reserve memory for vectors:
					tempVertices.reserve(attrib.vertices.size() / 3);
					tempIndices.reserve(attrib.vertices.size() / 3);

					for (const auto& material : materials) {
						object.texture.diffuseTexturePath = mtl_basepath + material.diffuse_texname; //only diffuse texture is needed for now
						break;
					}
					for (const auto& shape : shapes) {
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
							// Check if vertex is unique and add it to the map if it is:
							if (uniqueVertices.count(vertex) == 0) {
								uniqueVertices[vertex] = static_cast<uint32_t>(tempVertices.size());
								tempVertices.push_back(std::move(vertex));
							}
							tempIndices.push_back(uniqueVertices[vertex]);
						}
					}

					// batch insert vertices and indices into object:
					modelMtx.lock();
					object.vertices.insert(object.vertices.end(), tempVertices.begin(), tempVertices.end());
					object.indices.insert(object.indices.end(), tempIndices.begin(), tempIndices.end());
					object.isLoaded = true;
					modelMtx.unlock();
					}));
			}
		}
		for (auto& t : threads) {
			t.join();
		}

		for (auto& object : objects) {
			if (object.isLoaded) {
				debugStruct(object);
			}
		}
	}
	void debugStruct(model stru) {
		std::cout << " ----------------" << std::endl;
		std::cout << "model: " << stru.pathObj << std::endl;
		std::cout << "vertices: " << stru.vertices.size() << std::endl;
		std::cout << "indices: " << stru.indices.size() << std::endl;
		std::cout << "texture: " << stru.texture.diffuseTexturePath << std::endl;
		std::cout << " ----------------" << std::endl;
	}

	void createInstance() {
		VkApplicationInfo info{};
		info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // VK_STRUCTURE_TYPE_APPLICATION_INFO is a constant that tells Vulkan which structure you are using, which allows the implementation to read the data accordingly
		info.pApplicationName = "My Engine"; //the "p" is a naming convention that indicates a pointer to a null-terminated string
		info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		info.pEngineName = "No Engine";
		info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		info.apiVersion = VK_API_VERSION_1_0;
		VkInstanceCreateInfo newInfo{};
		newInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		newInfo.pApplicationInfo = &info;
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		newInfo.enabledExtensionCount = glfwExtensionCount;
		newInfo.ppEnabledExtensionNames = glfwExtensions;
		newInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		newInfo.ppEnabledLayerNames = validationLayers.data();

		if (vkCreateInstance(&newInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance! " + resultStr(vkCreateInstance(&newInfo, nullptr, &instance)));
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
			if (isDeviceSuitableG(device) && isDeviceSuitableP(device, surface)) { //isDeviceSuitableG checks if the device is suitable for graphics, isDeviceSuitableP checks if the device is suitable for presentation
				std::cout << "GPU and Presentation device found!" << std::endl;
				physicalDevice = device;
				break;
			}
		}
		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU for graphics and presentation");
		}
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamiliesG(physicalDevice);
		float queuePriority = 1.0f;
		VkDeviceQueueCreateInfo queueInf{};
		queueInf.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; //creates a structure to hold que family info
		queueInf.queueFamilyIndex = indices.graphicsFamily.value(); // index of the queue family to create gotten from the findQueueFamilies function
		queueInf.queueCount = 1;
		queueInf.pQueuePriorities = &queuePriority;
		VkPhysicalDeviceFeatures deviceFeatures{}; //this struct is used to specify the features we will be using. such as geometry shaders, anisotropic filtering, etc.
		VkDeviceCreateInfo newinfo{}; //specify which queues to create
		newinfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		newinfo.pQueueCreateInfos = &queueInf; //queues to create
		newinfo.queueCreateInfoCount = 1;
		newinfo.pEnabledFeatures = &deviceFeatures; //device features to enable
		const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME }; // specify the device extensions to enable
		newinfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		newinfo.ppEnabledExtensionNames = deviceExtensions.data();
		newinfo.enabledLayerCount = 0;
		newinfo.ppEnabledLayerNames = nullptr;
		if (vkCreateDevice(physicalDevice, &newinfo, nullptr, &device) != VK_SUCCESS) { // if logic device creation fails, output error
			throw std::runtime_error("failed to create logical device! " + resultStr(vkCreateDevice(physicalDevice, &newinfo, nullptr, &device)));
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
	//
	void setupDepthResources() {
		depthFormat = findDepthFormat();
		createDepthImage();
		depthImageView = createDepthView();
	}
	void createDepthImage() { //create the depth image and allocate memory for it
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;  //2d image
		imageInfo.extent.width = swapChainExtent.width;
		imageInfo.extent.height = swapChainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1; //no mip mapping for now lol
		imageInfo.arrayLayers = 1;
		imageInfo.format = depthFormat;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; //intiial layout doesn't matter
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; //dstencil attachment
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT; //no multisampling
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; //only one queue family can access the image at a time

		if (vkCreateImage(device, &imageInfo, nullptr, &depthImage) != VK_SUCCESS) {
			throw std::runtime_error("failed to create depth image!");
		}
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, depthImage, &memRequirements);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &allocInfo, nullptr, &depthImageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate depth image memory!");
		}
		vkBindImageMemory(device, depthImage, depthImageMemory, 0);
	}
	VkImageView createDepthView() {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = depthFormat;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT; // only depth aspect of the image
		viewInfo.subresourceRange.baseMipLevel = 0; //no mip mapping
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView depthView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &depthView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create depth image view!");
		}

		return depthView;
	}
	void createImageViews() { //create the image views for the swap chain images
		swapChainImageViews.resize(swapChainImages.size()); // resize swapChainImageViews to hold all the image views
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo newinfo{};
			newinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			newinfo.image = swapChainImages[i]; // assign the current swap chain image
			newinfo.viewType = VK_IMAGE_VIEW_TYPE_2D; // 2d image for now
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

	void createUBOs() {
		uboBuffers.resize(objects.size());
		uboBufferMemories.resize(objects.size());
		for (size_t i = 0; i < objects.size(); i++) {
			VkBufferCreateInfo bufferCreateInfo{};
			bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferCreateInfo.size = sizeof(UniformBufferObject);
			bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
			bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &uboBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create UBO buffer!");
			}

			VkMemoryRequirements memoryRequirements;
			vkGetBufferMemoryRequirements(device, uboBuffers[i], &memoryRequirements);

			VkMemoryAllocateInfo allocateInfo{};
			allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocateInfo.allocationSize = memoryRequirements.size;
			allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); //find a memory type that is host visible and coherent

			if (vkAllocateMemory(device, &allocateInfo, nullptr, &uboBufferMemories[i]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate memory for UBO buffer!");
			}
			vkBindBufferMemory(device, uboBuffers[i], uboBufferMemories[i], 0); // bind the buffer to the memory
		}
	}
	void calcMatrixes(model& o) {
		convertMatrix(forms::mat4::modelMatrix(o.position, o.rotation, o.scale), o.modelMatrix);
		convertMatrix(forms::mat4::viewMatrix(cam.camPos, cam.camAngle), o.viewMatrix);
		convertMatrix(forms::mat4::perspective(60.0f, swapChainExtent.width / static_cast<float>(swapChainExtent.height), 0.1f, 1000.0f), o.projectionMatrix);
		o.projectionMatrix[5] *= -1; //flip the y for vulkan
	}

	void updateUBO(const camData& cam) { // needs optimization later
		for (size_t i = 0; i < objects.size(); i++) {
			calcMatrixes(objects[i]);

			// create and populate the UBO:
			UniformBufferObject ubo;
			memcpy(ubo.model, objects[i].modelMatrix, sizeof(objects[i].modelMatrix));
			memcpy(ubo.view, objects[i].viewMatrix, sizeof(objects[i].viewMatrix));
			memcpy(ubo.proj, objects[i].projectionMatrix, sizeof(objects[i].projectionMatrix));

			// transfer the ubo struct into the buffer:
			void* data;
			vkMapMemory(device, uboBufferMemories[i], 0, sizeof(ubo), 0, &data);
			memcpy(data, &ubo, sizeof(ubo));
			vkUnmapMemory(device, uboBufferMemories[i]);
		}
	}
	void convertMatrix(const forms::mat4& source, float destination[16]) { //converts a 4x4 matrix to a flat array for vulkan
		int index = 0;
		for (int column = 0; column < 4; column++) {
			for (int row = 0; row < 4; row++) {
				destination[index] = source.m[row][column];
				index++;
			}
		}
	}
	forms::mat4 unflattenMatrix(const float source[16]) { //converts a flat array to a 4x4 matrix
		forms::mat4 destination;
		int index = 0;
		for (int column = 0; column < 4; column++) {
			for (int row = 0; row < 4; row++) {
				destination.m[row][column] = source[index];
				index++;
			}
		}
		return destination;
	}

	void createDSLayout() {
		std::array<VkDescriptorSetLayoutBinding, 2> bindings{};

		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT; //UBO will be accessed from the vertex shader

		bindings[1].binding = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // combned image sampler will be accessed from the fragment shader

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor set layout!");
		}
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

	void createDSPool() {
		std::array<VkDescriptorPoolSize, 2> poolSizes{};

		// matrix ubo descriptor set:
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(objects.size());

		// texture data descriptor set:
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(objects.size());

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(objects.size());

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor pool!");
		}
	}


	void createDS() {
		VkDescriptorBufferInfo bufferInfo{}; //info about the UBO
		createUBOs();
		VkDescriptorImageInfo imageInfo;
		descriptorSets.resize(objects.size());
		std::vector<std::thread> threads;

		for (int i = 0; i < objects.size(); i++) {
			threads.push_back(std::thread(&Engine::createTexturedImage, this, objects[i].texture.diffuseTexturePath, std::ref(objects[i])));
		}

		for (auto& thread : threads) {
			if (thread.joinable()) {
				thread.join();
			}
		}
		for (int i = 0; i < objects.size(); i++) {
			createTextureImgView(objects[i]); // create the texture image view
			createTS(objects[i]); // create the texture sampler
			bufferInfo.buffer = uboBuffers[i];
			bufferInfo.offset = 0; //offset in the buffer where the UBO starts
			bufferInfo.range = sizeof(UniformBufferObject);
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = objects[i].texture.textureImageView;
			imageInfo.sampler = objects[i].texture.textureSampler;

			VkDescriptorSetLayout layouts[] = { descriptorSetLayout };
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.descriptorSetCount = 1;
			allocInfo.pSetLayouts = layouts;

			if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets[i]) != VK_SUCCESS) { // allocate the descriptor set for each object
				throw std::runtime_error("Failed to allocate descriptor set!");
			}

			std::array<VkWriteDescriptorSet, 2> descriptorWrites{}; // vector to hold the info about the UBO and the texture sampler

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; //type=UBO
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; //type=combined image sampler
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	void setupDescriptorSets() {
		createDSLayout();
		createDSPool();
		createDS(); //create the descriptor set
	}
	void realtimeLoad(std::string p) {
		model m = objects[0];
		forms::mat4 test;
		// modify the copied model
		forms::vec3 worldPos = { 0.0f, 0.0f, 0.0f };
		worldPos = forms::mat4::inverseMatrix(unflattenMatrix(m.modelMatrix)).vecMatrix(cam.camPos);
		m.scale = { 0.01f, 0.01f, 0.01f };
		m.position = worldPos.multiply(10, 10, 10);
		m.startObj = false;
		objects.push_back(m);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		setupDescriptorSets();
	}
	void printMatrix(forms::mat4 mat) {
		for (int row = 0; row < 4; ++row) {
			for (int col = 0; col < 4; ++col) {
				std::cout << mat.m[row][col] << "\t";
			}
			std::cout << std::endl;
		}
	}

	void createTS(model& m) { //create texture sampler
		VkSamplerCreateInfo samplerInf{}; // create sampler info
		samplerInf.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInf.magFilter = VK_FILTER_LINEAR; // magnification filter
		samplerInf.minFilter = VK_FILTER_LINEAR; // minification filter
		samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT; // repeat the texture when out of bounds (horizontal)
		samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT; // (vertical)
		samplerInf.anisotropyEnable = VK_FALSE; // warps textures to fit objects, etc
		samplerInf.maxAnisotropy = 16;
		samplerInf.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInf.unnormalizedCoordinates = VK_FALSE; // enable normalized coordinates
		samplerInf.compareEnable = VK_FALSE; // compare enable (for shadow mapping)
		samplerInf.compareOp = VK_COMPARE_OP_ALWAYS; // comparison operation result is always VK_TRUE
		samplerInf.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInf.minLod = 0.0f;
		samplerInf.maxLod = static_cast<float>(m.texture.mipLevels); // set the maximum level-of-detail value to the number of mip levels
		if (vkCreateSampler(device, &samplerInf, nullptr, &m.texture.textureSampler) != VK_SUCCESS) { // create sampler
			throw std::runtime_error("failed to create texture sampler!");
		}
	}
	void createTextureImgView(model& m) {
		VkImageViewCreateInfo viewInf{};
		viewInf.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInf.image = m.texture.textureImage;
		viewInf.viewType = VK_IMAGE_VIEW_TYPE_2D; //view type is 3D
		viewInf.format = VK_FORMAT_R8G8B8A8_SRGB; //rgba
		viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // set aspect mask to color bit
		viewInf.subresourceRange.baseMipLevel = 0;
		viewInf.subresourceRange.levelCount = m.texture.mipLevels - viewInf.subresourceRange.baseMipLevel; //miplevel is influenced by the base 
		viewInf.subresourceRange.baseArrayLayer = 0;
		viewInf.subresourceRange.layerCount = 1;
		if (vkCreateImageView(device, &viewInf, nullptr, &m.texture.textureImageView) != VK_SUCCESS) {
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
	void createStagingBuffer(model& m) { // buffer to transfer data from the CPU (imageData) to the GPU sta
		VkBufferCreateInfo bufferInf{};
		bufferInf.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInf.size = imageSize;
		bufferInf.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufferInf.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (vkCreateBuffer(device, &bufferInf, nullptr, &m.stagingBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create staging buffer!");
		}
		// get mem requirements;
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, m.stagingBuffer, &memRequirements);
		VkMemoryAllocateInfo allocInf{};
		allocInf.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInf.allocationSize = memRequirements.size;
		allocInf.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (vkAllocateMemory(device, &allocInf, nullptr, &m.stagingBufferMem) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate staging buffer memory!");
		}

		// bind the memory to the buffer:
		vkBindBufferMemory(device, m.stagingBuffer, m.stagingBufferMem, 0);
		void* data;
		if (vkMapMemory(device, m.stagingBufferMem, 0, imageSize, 0, &data) != VK_SUCCESS) {
			throw std::runtime_error("failed to map staging buffer memory!");
		}
		memcpy(data, imageData, imageSize);
		vkUnmapMemory(device, m.stagingBufferMem);
	}

	void createTexturedImage(std::string path, model& m) {
		// reset the staging buffer and memory:
		if (m.stagingBuffer == VK_NULL_HANDLE) {
			getImageData(path);
			createStagingBuffer(m);
		}
		{
			std::lock_guard<std::mutex> lock(descMtx); //when the thread is done, unlock the mutex
			m.texture.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
			// create image:
			VkImageCreateInfo imageInf{};
			imageInf.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInf.imageType = VK_IMAGE_TYPE_2D;
			imageInf.extent.width = texWidth;
			imageInf.extent.height = texHeight;
			imageInf.extent.depth = 1;
			imageInf.mipLevels = m.texture.mipLevels;
			imageInf.arrayLayers = 1;
			imageInf.format = VK_FORMAT_R8G8B8A8_SRGB;
			imageInf.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInf.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInf.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
			// transfer destination bit (copy from buffer to image) and sampled bit (can be used for sampling operations)

			imageInf.samples = VK_SAMPLE_COUNT_1_BIT; // no multisampling
			imageInf.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			if (vkCreateImage(device, &imageInf, nullptr, &m.texture.textureImage) != VK_SUCCESS) {
				throw std::runtime_error("failed to create texture image!");
			}

			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(device, m.texture.textureImage, &memRequirements);

			VkMemoryAllocateInfo allocInf{};
			allocInf.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInf.allocationSize = memRequirements.size;
			allocInf.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			if (vkAllocateMemory(device, &allocInf, nullptr, &m.texture.textureImageMemory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate texture image memory!!!");
			}
			vkBindImageMemory(device, m.texture.textureImage, m.texture.textureImageMemory, 0); // bind the memory to the image through TIM (texture image memory) and the image

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

			VkCommandBuffer tempBuffer = beginSingleTimeCommands();

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; //specifies the layout to transition from
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; //specifies the layout to transition to
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; /// TODO
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = m.texture.textureImage;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = m.texture.mipLevels;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; //specifies the operations that must be finished on the old layout before it transitions to the new layout
			vkCmdPipelineBarrier(tempBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier); //from the top of the pipeline to the transfer stage

			vkCmdCopyBufferToImage(tempBuffer, m.stagingBuffer, m.texture.textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region); //copy the data from the staging buffer to the image

			int mipWidth = texWidth;
			int mipHeight = texHeight;

			for (uint32_t i = 0; i < m.texture.mipLevels; i++) {
				VkImageMemoryBarrier barrierToSrc{};
				barrierToSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
				barrierToSrc.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				barrierToSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrierToSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrierToSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrierToSrc.image = m.texture.textureImage;
				barrierToSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; //this means that the image is a color image not a depth or stencil image
				barrierToSrc.subresourceRange.baseMipLevel = i;
				barrierToSrc.subresourceRange.levelCount = 1;
				barrierToSrc.subresourceRange.baseArrayLayer = 0;
				barrierToSrc.subresourceRange.layerCount = 1;
				barrierToSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; //specifies the operations that must be finished on the old layout before it transitions to the new layout
				barrierToSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				vkCmdPipelineBarrier(tempBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrierToSrc); //from the top of the pipeline to the transfer stage

				//iterate through all but the last mip level to blit from the previous mip level to the next mip level:
				if (i < m.texture.mipLevels - 1) {
					VkImageBlit blit{};
					blit.srcOffsets[0] = { 0, 0, 0 };
					blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
					blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					blit.srcSubresource.mipLevel = i;
					blit.srcSubresource.baseArrayLayer = 0;
					blit.srcSubresource.layerCount = 1;
					blit.dstOffsets[0] = { 0, 0, 0 };
					blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 }; //if the mip level is greater than 1, divide by 2, otherwise set to 1
					blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					blit.dstSubresource.mipLevel = i + 1;
					blit.dstSubresource.baseArrayLayer = 0;
					blit.dstSubresource.layerCount = 1;
					vkCmdBlitImage(tempBuffer, m.texture.textureImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m.texture.textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
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
			endSingleTimeCommands(tempBuffer);
			stbi_image_free(imageData);
			imageData = nullptr;
		}
	}
	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; //specifies if the command buffer is primary or secondary
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;
		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; //one time command buffer
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE); //submit the command buffer to the queue
		vkQueueWaitIdle(graphicsQueue); //wait for the queue to be idle
		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer); //free the command buffer
	}

	void setupShaders() {
		std::vector<char> vertShaderCode = readFile("vertex_shader.spv"); //read the vertex shader binary
		std::vector<char> fragShaderCode = readFile("fragment_shader.spv");
		vertShaderModule = createShaderModule(vertShaderCode);
		fragShaderModule = createShaderModule(fragShaderCode);
	}

	static void check_vk_result(VkResult err) { //used to debug imgui errors that have to do with vulkan 
		if (err == 0)
			return;
		std::cerr << "VkResult is " << err << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
		assert(err == 0); //if true, continue, if false, throw error
	}
	void createGraphicsPipelineOpaque() { //pipeline for ONLY opaque objects
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

		std::array<VkVertexInputAttributeDescription, 4> attrDesc; // attr0 is position, attr1 is color, attr2 is alpha, attr3 is texture coordinates

		attrDesc[0].binding = 0;
		attrDesc[0].location = 0;
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for position
		attrDesc[0].offset = offsetof(Vertex, pos);

		attrDesc[1].binding = 0;
		attrDesc[1].location = 1;
		attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT; // 3 floats for color
		attrDesc[1].offset = offsetof(Vertex, col);

		attrDesc[2].binding = 0;
		attrDesc[2].location = 2;
		attrDesc[2].format = VK_FORMAT_R32_SFLOAT; // 1 float for alpha
		attrDesc[2].offset = offsetof(Vertex, alpha);

		attrDesc[3].binding = 0;
		attrDesc[3].location = 3;
		attrDesc[3].format = VK_FORMAT_R32G32_SFLOAT; // 2 floats for texture coordinates
		attrDesc[3].offset = offsetof(Vertex, tex);


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
		vp.width = (float)swapChainExtent.width; //width for the viewport is the swap chain extent widthh
		vp.height = (float)swapChainExtent.height;
		vp.minDepth = 0.0f;
		vp.maxDepth = 1.0f;
		VkRect2D scissor{};
		scissor.offset = { 0, 0 }; //0,0 offset (top left corner)
		scissor.extent = swapChainExtent;
		VkPipelineViewportStateCreateInfo vpState{}; //create a struct for the viewport state
		vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; //assign the struct type to the viewport state
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
		dynamicState.dynamicStateCount = std::size(dynamicStates);
		dynamicState.pDynamicStates = dynamicStates;

		//pipeline layout setup: Allows for uniform variables to be passed into the shader. no uniform variables are used yet thats fior later

		VkPipelineLayoutCreateInfo pipelineLayoutInf{};
		pipelineLayoutInf.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInf.setLayoutCount = 1; //number of descriptor set layouts
		pipelineLayoutInf.pSetLayouts = &descriptorSetLayout; //array of descriptor set layouts (a pointer to the array)
		pipelineLayoutInf.pushConstantRangeCount = 0; //number of push constant ranges
		pipelineLayoutInf.pPushConstantRanges = nullptr; //array of push constant ranges
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

		// imgui initialization:
		uint32_t graphicsQueueFamily = findQueueFamiliesG(physicalDevice).graphicsFamily.value();
		//implement here

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
		std::cout << "Opaque Graphics Pipeline Created Successfully!" << std::endl;

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

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamiliesG(physicalDevice);
		VkCommandPoolCreateInfo poolInf{};
		poolInf.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInf.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); //the queue family that will be using this command pool
		poolInf.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // enable reset command buffer flag
		VkResult result = vkCreateCommandPool(device, &poolInf, nullptr, &commandPool);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool! " + resultStr(result));
		}
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
	void createVertexBuffer() {
		vertBuffers.resize(objects.size());
		vertBufferMems.resize(objects.size());
		for (size_t i = 0; i < objects.size(); i++) {
			VkDeviceSize bufferSize = sizeof(Vertex) * objects[i].vertices.size(); //size of the buffer. formula is: size of the data * number of vertices
			VkBufferCreateInfo bufferInf{};
			bufferInf.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInf.size = bufferSize; //size of the buffer
			bufferInf.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; //buffer will be used as a vertex buffer
			bufferInf.sharingMode = VK_SHARING_MODE_EXCLUSIVE; //buffer will be exclusive to a single queue family at a time
			if (vkCreateBuffer(device, &bufferInf, nullptr, &vertBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create vertex buffer!");
			}
			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(device, vertBuffers[i], &memRequirements); //get the memory requirements for the vertex buffer
			VkMemoryAllocateInfo allocInf{}; //struct to hold memory allocation info
			allocInf.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInf.allocationSize = memRequirements.size;

			//params are: memory requirements, properties of the memory, and the memory type we are looking for
			allocInf.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); //bitwise OR the memory properties to find the memory type
			if (vkAllocateMemory(device, &allocInf, nullptr, &vertBufferMems[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate vertex buffer memory!");
			}
			vkBindBufferMemory(device, vertBuffers[i], vertBufferMems[i], 0); //bind the vertex buffer to the vertex buffer memory
			void* data;
			vkMapMemory(device, vertBufferMems[i], 0, bufferSize, 0, &data);
			memcpy(data, objects[i].vertices.data(), bufferSize);
			vkUnmapMemory(device, vertBufferMems[i]);
		}
	}
	void createIndexBuffer() {
		indBuffers.resize(objects.size());
		indBufferMems.resize(objects.size());
		for (size_t i = 0; i < objects.size(); i++) {
			VkDeviceSize bufferSize = sizeof(uint32_t) * objects[i].indices.size(); //size of the buffer. formula is: size of the data * number of indices
			VkBufferCreateInfo bufferInf{};
			bufferInf.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInf.size = bufferSize; //size of the buffer
			bufferInf.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT; //buffer will be used as a index buffer
			bufferInf.sharingMode = VK_SHARING_MODE_EXCLUSIVE; //buffer will be exclusive to a single queue family at a time
			if (vkCreateBuffer(device, &bufferInf, nullptr, &indBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create index buffer!");
			}
			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(device, indBuffers[i], &memRequirements); //get the memory requirements for the index buffer
			VkMemoryAllocateInfo allocInf{}; //struct to hold memory allocation info
			allocInf.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInf.allocationSize = memRequirements.size;

			//params are: memory requirements, properties of the memory, and the memory type we are looking for
			allocInf.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); //bitwise OR the memory properties to find the memory type
			if (vkAllocateMemory(device, &allocInf, nullptr, &indBufferMems[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate index buffer memory!");
			}
			vkBindBufferMemory(device, indBuffers[i], indBufferMems[i], 0); //bind the index buffer to the index buffer memory
			void* data;
			vkMapMemory(device, indBufferMems[i], 0, bufferSize, 0, &data);
			memcpy(data, objects[i].indices.data(), bufferSize);
			vkUnmapMemory(device, indBufferMems[i]);
		}
	}

	void recordCommandBuffers() { //records and submits the command buffers
		for (size_t i = 0; i < commandBuffers.size(); i++) {
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
			beginInfo.pInheritanceInfo = nullptr; //if nullptr, then it is a primary command buffer
			vkResetCommandBuffer(commandBuffers[i], 0);
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

			std::array<VkClearValue, 2> clearValues = {
		VkClearValue{0.68f, 0.85f, 0.90f, 1.0f},  // clear color: light blue
		VkClearValue{1.0f, 0}  // clear depth: 1.0f, 0 stencil
			};

			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline); // bind the graphics pipeline to the command buffer

			for (size_t j = 0; j < objects.size(); j++) {
				// bind the descriptor set for each object (each set is a different object):
				VkDescriptorSet dSets[] = { descriptorSets[j] };
				vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, dSets, 0, nullptr);
				if (j >= vertBuffers.size()) {
					std::cerr << "Warning: missing vertex buffer for object " << j + 1 << std::endl;
					continue;
				}

				VkBuffer vertexBuffersArray[] = { vertBuffers[j] };
				VkDeviceSize offsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffersArray, offsets);

				if (j >= indBuffers.size()) {
					std::cerr << "Warning: missing index buffer for object " << j + 1 << std::endl;
					continue;
				}

				VkBuffer indexBuffersArray[] = { indBuffers[j] };
				vkCmdBindIndexBuffer(commandBuffers[i], indexBuffersArray[0], 0, VK_INDEX_TYPE_UINT32);

				// ensure object size is correct:
				uint32_t objectVertexCount = static_cast<uint32_t>(objects[j].vertices.size());
				if (objectVertexCount == 0) {
					std::cerr << "Warning: object " << j + 1 << " has an invalid size" << std::endl;
					continue;
				}

				vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(objects[j].indices.size()), 1, 0, 0, 0);
			}
			vkCmdEndRenderPass(commandBuffers[i]);
			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
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
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		setupDescriptorSets();
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);
		setupDepthResources();
		createGraphicsPipelineOpaque();
		createFrameBuffer();
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
	void drawF() { //draw frame function
		uint32_t imageIndex;
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &inFlightFences[currentFrame]);
		VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex); //acquire an image from the swap chain
		if (result == VK_ERROR_OUT_OF_DATE_KHR) { //fix
			vkDeviceWaitIdle(device);
			recreateSwap();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image! " + resultStr(result));
		}
		//submit the command buffer:
		VkSubmitInfo submitInf{};
		submitInf.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore }; //semaphore to wait on before execution begins
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; //stage to wait: color attachment output stage
		submitInf.waitSemaphoreCount = 1;
		submitInf.pWaitSemaphores = waitSemaphores;
		submitInf.pWaitDstStageMask = waitStages;
		submitInf.commandBufferCount = 1;
		submitInf.pCommandBuffers = &commandBuffers[imageIndex];
		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore }; //semaphore to signal when command buffer finishes execution
		submitInf.signalSemaphoreCount = 1; //number of semaphores to signal
		submitInf.pSignalSemaphores = signalSemaphores; //list of semaphores to signal
		if (vkQueueSubmit(graphicsQueue, 1, &submitInf, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}
		//present the image:
		VkPresentInfoKHR presentInf{};
		presentInf.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInf.waitSemaphoreCount = 1;
		presentInf.pWaitSemaphores = signalSemaphores;
		VkSwapchainKHR swapChains[] = { swapChain };
		presentInf.swapchainCount = 1; //number of swap chains to present to
		presentInf.pSwapchains = swapChains; //list of swap chains to present to
		presentInf.pImageIndices = &imageIndex; //index of image in swap chain to present
		presentInf.pResults = nullptr; //optional array to receive results of each swap chain's presentation
		result = vkQueuePresentKHR(presentQueue, &presentInf);
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

	void recreateObjectBuffers() {
		vkDeviceWaitIdle(device);  // wait for all frames to finish
		for (auto vertBuffer : vertBuffers) {
			vkDestroyBuffer(device, vertBuffer, nullptr);
		}
		for (auto vertBufferMemory : vertBufferMems) {
			vkFreeMemory(device, vertBufferMemory, nullptr);
		}
		for (auto indexBuffer : indBuffers) {
			vkDestroyBuffer(device, indexBuffer, nullptr);
		}
		for (auto indexBufferMemory : indBufferMems) {
			vkFreeMemory(device, indexBufferMemory, nullptr);
		}
		createIndexBuffer();
		createVertexBuffer();
		recordCommandBuffers();  // re-record command buffers to reference the new buffers
	}
	void drawText(const char* text, float x, float y) {
		ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Always);

		ImGui::Begin("TextWindow", nullptr,
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoScrollbar |
			ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoInputs |
			ImGuiWindowFlags_NoBackground |
			ImGuiWindowFlags_NoBringToFrontOnFocus);

		ImGui::TextUnformatted(text);
		ImGui::End();
	}


	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawF();
			currentFrame = (currentFrame + 1) % swapChainImages.size();
			handleKeyboardInput(window); //handle keyboard input to change cam position
			recreateObjectBuffers();
			updateUBO(cam); //update ubo matricies and populate the buffer
		}
		vkDeviceWaitIdle(device);
	}
	void handleKeyboardInput(GLFWwindow* window) {
		float cameraSpeed = 0.004f; // Adjust the speed as needed. on the laptop 0.012f is good
		float cameraRotationSpeed = 0.4f;

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
		createInstance();
		createSurface();
		pickDevice();
		createLogicalDevice();
		initQueues(); //sets the queue family indices such as graphics and presentation
		loadModels(); //load the model data from the obj file
		createSC(); //create swap chain
		setupFences();
		createSemaphores();
		createCommandPool();
		createVertexBuffer();
		createIndexBuffer();
		setupShaders(); //read the shader files and create the shader modules
		setupDescriptorSets();
		setupDepthResources();
		createGraphicsPipelineOpaque();
		createFrameBuffer();
		createCommandBuffer();
		recordCommandBuffers(); //record and submit the command buffers (includes code for binding the descriptor set)
		std::cout << "Vulkan initialized successfully!" << std::endl;
	}
	void cleanup() { //FIX
		// destroy resources in reverse order of creation
		cleanupTextures(); //cleanup texture, descriptor and all sampler data
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
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		// clean up vertex buffer and its memory
		for (int i = 0; i < vertBuffers.size(); i++) {
			vkDestroyBuffer(device, vertBuffers[i], nullptr);
		}
		for (int i = 0; i < vertBufferMems.size(); i++) {
			vkFreeMemory(device, vertBufferMems[i], nullptr);
		}
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
	void cleanupTextures() { // cleanup textures, samplers and descriptors
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
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
	// 18. mip mapping and optimizations
	// 19. lighting
	// 20. shadows
	// 21. skybox
	// 22. collision detection
};
int main() {
	objects[0].pathObj = "models/gear/Gear1.obj";
	objects[0].rotation = { 0.0f, 70.0f, 0.0f };
	objects[1].pathObj = "models/gear2/Gear2.obj";
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