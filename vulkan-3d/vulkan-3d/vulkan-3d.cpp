#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <optional> //allows to create optional values in a more efficient way than using pointers
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <array>
const uint32_t WIDTH = 3200;
const uint32_t HEIGHT = 1800;
struct Vertex {
	float posX;
	float posY;
	float colR;
	float colG;
	float colB;
};
struct testVelo {
	double vx;
	double vy;
};
std::vector<testVelo> velocities = {
	{0.001, 0},
	{0.001, 0},
	{0.001, 0}
};

std::vector<Vertex> triangle1vert = {
	{-1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
	{-0.3f, -1.0f, 0.0f, 1.0f, 0.0f},
	{0.3f, -0.8f, 1.0f, 0.0f, 0.0f}
};
std::vector<Vertex> triangle2vert = {
	{-0.2f, 0.0f, 0.0f, 0.0f, 1.0f},
	{-0.3f, -1.0f, 0.0f, 1.0f, 0.0f},
	{0.0f, -1.0f, 1.0f, 0.0f, 0.0f}
};
std::vector<Vertex> triangle3vert = {
	{-0.8f, 0.0f, 0.0f, 0.0f, 1.0f},
	{-0.3f, -0.8f, 0.0f, 1.0f, 0.0f},
	{0.3f, -0.8f, 1.0f, 0.0f, 0.0f}
};

std::vector<std::vector<Vertex>>objects = { triangle1vert, triangle2vert, triangle3vert };

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
	std::vector<VkImageView> swapChainImageViews;
	VkViewport vp{};
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
	VkQueue presentQueue;
	VkQueue graphicsQueue;

	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}
	const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
	};

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
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1; //the number of images is based on the minimum number of images plus one
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
		newinfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; //ignore the alpha channel
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
	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size()); // resize swapChainImageViews to hold all the image views
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo newinfo{};
			newinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			newinfo.image = swapChainImages[i]; // assign the current swap chain image
			newinfo.viewType = VK_IMAGE_VIEW_TYPE_2D; // set the image view type to 2D
			newinfo.format = swapChainImageFormat;
			newinfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
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
	void setupGraphicsPipeline() {
		std::vector<char> vertShaderCode = readFile("vertex_shader.spv"); //read the vertex shader binary
		std::vector<char> fragShaderCode = readFile("fragment_shader.spv");
		vertShaderModule = createShaderModule(vertShaderCode);
		fragShaderModule = createShaderModule(fragShaderCode);
	}
	void createGraphicsPipeline() {
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

		//vertex input setup (tells vulkan how to read/organize vertex data based on the stride, offset, and rate)
		VkVertexInputBindingDescription bindDesc{};
		bindDesc.binding = 0;
		bindDesc.stride = sizeof(Vertex); //num of bytes from one entry to the next
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; //the rate when data is loaded

		std::array<VkVertexInputAttributeDescription, 2> attrDesc; //attr0 is position, attr1 is color

		attrDesc[0].binding = 0;
		attrDesc[0].location = 0;
		attrDesc[0].format = VK_FORMAT_R32G32_SFLOAT;
		attrDesc[0].offset = offsetof(Vertex, posX);

		attrDesc[1].binding = 0;
		attrDesc[1].location = 1;
		attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attrDesc[1].offset = offsetof(Vertex, colR);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{}; //vertex input state struct
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;  //assign the struct type to the vertex input state
		vertexInputInfo.vertexBindingDescriptionCount = 1;  //value is set to the amount of binding descriptions
		vertexInputInfo.pVertexBindingDescriptions = &bindDesc;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
		vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data(); //assign the vertex input attribute descriptions


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

		//rasterizer setup: Transforms 3D primitives into 2D fragments for display on the screen
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE; //if true, fragments that are beyond the near and far planes are clamped
		rasterizer.rasterizerDiscardEnable = VK_FALSE; //if true, geometry never passes through the rasterizer
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //fill the area of the poly with fragments
		rasterizer.lineWidth = 1.0f; //thickness of fragment lines
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; //cull the back faces of triangle
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_TRUE; //if false, no depth bias is applied to fragments
		rasterizer.depthBiasConstantFactor = 0.0f; //const value that is added to the depth value of a frag
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		//multisampling/anti-aliasing setup: Aggregates multiple samples per pixel, considering alpha values, color, and depth information, and outputs a single colored pixel
		VkPipelineMultisampleStateCreateInfo multiSamp{};
		multiSamp.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multiSamp.sampleShadingEnable = VK_FALSE; //if true, enable sample shading in the pipeline
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
		dStencil.back.failOp = VK_STENCIL_OP_KEEP;
		dStencil.back.passOp = VK_STENCIL_OP_KEEP;
		dStencil.back.depthFailOp = VK_STENCIL_OP_KEEP;
		dStencil.back.compareOp = VK_COMPARE_OP_ALWAYS;
		dStencil.back.compareMask = 0;
		dStencil.back.writeMask = 0;
		dStencil.back.reference = 0;


		//color blending setup: Combines the output of the fragment shader with the color that is already in the viewbuffer
		VkPipelineColorBlendAttachmentState colorBA{}; //color blend attachment struct
		colorBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; //color channels to apply the blending operation to
		colorBA.blendEnable = VK_TRUE; //enable blending
		colorBA.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; //blending factors for color channels
		colorBA.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; //dst is the color already in the framebuffer and src is the color being output from the fragment shader
		colorBA.colorBlendOp = VK_BLEND_OP_ADD; //blending operation to perform
		colorBA.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; //blending factors for alpha channel
		colorBA.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBA.alphaBlendOp = VK_BLEND_OP_ADD;
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
		pipelineLayoutInf.setLayoutCount = 0; //number of descriptor sets used by the pipeline such as uniform variables
		pipelineLayoutInf.pSetLayouts = nullptr; //array of descriptor set layouts
		pipelineLayoutInf.pushConstantRangeCount = 0; //number of push constant ranges
		pipelineLayoutInf.pPushConstantRanges = nullptr; //array of push constant ranges
		VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInf, nullptr, &pipelineLayout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!! " + resultStr(result));
		}

		//render pass setup: Describes the attachments used by the pipeline and how many samples to use for each attachment
		//attachment: A memory location that can be read from or written to by a pipeline to perform rendering operations
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

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; //type of pipeline to bind to
		subpass.colorAttachmentCount = 1; //number of color attachments
		subpass.pColorAttachments = &colorAttachmentRef; //array of color attachment references

		//define the render pass
		VkRenderPassCreateInfo renderPassInf{};
		renderPassInf.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInf.attachmentCount = 1; //number of attachments
		renderPassInf.pAttachments = &colorAttachment; //array of attachments
		renderPassInf.subpassCount = 1; //number of subpasses
		renderPassInf.pSubpasses = &subpass; //array of subpasses
		VkResult RenderPassResult = vkCreateRenderPass(device, &renderPassInf, nullptr, &renderPass);
		if (RenderPassResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass! " + resultStr(RenderPassResult));
		}
		std::cout << " Render pass created successfully" << std::endl;

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
		pipelineInf.subpass = 0; // Index of the subpass where this graphics pipeline is to be used
		pipelineInf.basePipelineHandle = VK_NULL_HANDLE; // Optional: set later
		pipelineInf.basePipelineIndex = -1; // Optional: set later
		VkResult pipelineResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInf, nullptr, &graphicsPipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		std::cout << "Graphics Pipeline Created Successfully!" << std::endl;
	}
	void initVulkan() { //initializes Vulkan functions
		createInstance();
		createSurface();
		pickDevice();
		createLogicalDevice();
		initQueues(); //sets the queue family indices such as graphics and presentation
		createSC(); //create swap chain
		createCommandPool();
		createVertexBuffer();
		setupGraphicsPipeline();
		createGraphicsPipeline(); //create the graphics pipeline
		createFrameBuffer();
		createCommandBuffer();
		recordCommandBuffers();
		createSemaphores();
		std::cout << "Vulkan Initialized Successfully!" << std::endl;
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawF();
			updateObjects();
			recreateVertexBuffer();
		}
		vkDeviceWaitIdle(device);
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
			VkDeviceSize bufferSize = sizeof(objects[i][0]) * objects[i].size(); //size of the buffer. formula is: size of the data * number of vertices
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
			memcpy(data, objects[i].data(), bufferSize);
			vkUnmapMemory(device, vertBufferMems[i]);
		}
	}

	void recordCommandBuffers() {
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
			VkClearValue clearColor = { 0.68f, 0.85f, 0.90f, 1.0f }; //light blue
			renderPassInfo.clearValueCount = 1; // 1=clear value is a color, 2 = clear value is a depth/stencil buffer, 0 = no attachments to clear
			renderPassInfo.pClearValues = &clearColor;

			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline); // bind the graphics pipeline to the command buffer

			for (size_t j = 0; j < objects.size(); j++) {
				// ensure vertex buffer and object correspondence
				if (j >= vertBuffers.size()) {
					std::cerr << "Warning: missing vertex buffer for object " << j + 1 << std::endl;
					continue;
				}

				VkBuffer vertexBuffersArray[] = { vertBuffers[j] };
				VkDeviceSize offsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffersArray, offsets);

				// ensure object size is correct
				uint32_t objectVertexCount = static_cast<uint32_t>(objects[j].size());
				if (objectVertexCount == 0) {
					std::cerr << "Warning: object " << j + 1 << " has an invalid size" << std::endl;
					continue;
				}

				vkCmdDraw(commandBuffers[i], objectVertexCount, 1, 0, 0);
				std::cout << "Drawing object " << j + 1 << std::endl;
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
			VkImageView attachments[] = { swapChainImageViews[i] }; //array of attachments
			VkFramebufferCreateInfo framebufferInf{};
			framebufferInf.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInf.renderPass = renderPass; //render pass that this framebuffer will be compatible with
			framebufferInf.attachmentCount = 1; //attachment is the image from swap chain
			framebufferInf.pAttachments = attachments; //array of attachments
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
	void recreateSwap() {} //update later

	void drawF() { //draw frame function
		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex); //acquire an image from the swap chain
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwap();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image! " + resultStr(result));
		}
		VkSubmitInfo submitInf{};
		submitInf.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore }; //semaphore to wait on before execution begins
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; //pipeline stage to wait at
		submitInf.waitSemaphoreCount = 1;
		submitInf.pWaitSemaphores = waitSemaphores;
		submitInf.pWaitDstStageMask = waitStages; //list of pipeline stages to wait on
		submitInf.commandBufferCount = 1;
		submitInf.pCommandBuffers = &commandBuffers[imageIndex];
		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore }; //semaphore to signal when command buffer finishes execution
		submitInf.signalSemaphoreCount = 1; //number of semaphores to signal
		submitInf.pSignalSemaphores = signalSemaphores; //list of semaphores to signal
		if (vkQueueSubmit(graphicsQueue, 1, &submitInf, VK_NULL_HANDLE) != VK_SUCCESS) { //submit the command buffer to the graphics queue
			throw std::runtime_error("failed to submit draw command buffer!");
		}
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
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwap();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image! " + resultStr(result));
		}
		vkQueueWaitIdle(presentQueue); //wait for the queue to be idle before continuing
	}
	void recreateVertexBuffer() {
		for (auto vertBuffer : vertBuffers) {
			vkDestroyBuffer(device, vertBuffer, nullptr);
		}
		for (auto vertBufferMemory : vertBufferMems) {
			vkFreeMemory(device, vertBufferMemory, nullptr);
		}
		createVertexBuffer();
	}
	void updateObjects() {
		for (size_t i = 0; i < objects.size(); i++) {
			for (Vertex& vertex : objects[i]) { //move the objects
				vertex.posX += velocities[i].vx;
				vertex.posY += velocities[i].vy;
			}
		}
	}

	void cleanup() {
		// destroy resources in reverse order of creation
		vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
		vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		for (auto frameBuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, frameBuffer, nullptr);
		}
		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);

		// clean up vertex buffer and its memory
		for (int i = 0; i < vertBuffers.size(); i++) {
			vkDestroyBuffer(device, vertBuffers[i], nullptr);
		}
		for (int i = 0; i < vertBufferMems.size(); i++) {
			vkFreeMemory(device, vertBufferMems[i], nullptr);
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
	// 12. moving objects
	// 13. fences
	// 14. convert to 3d
	// 15. textures
	// 16. shadows
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