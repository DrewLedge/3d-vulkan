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
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	VkShaderModule fragShader;
	VkShaderModule vertShader;
	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
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
		newInfo.enabledLayerCount = 0;

		if (vkCreateInstance(&newInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}
	void pickDevice() { //outputs number of devices that support Vulkan
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);//outputs the number of devices that support Vulkanm into deviceCount
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()); // stores the physical devices into the devices vector
		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}
		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU for graphics");
		}
	}
	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		float queuePriority = 1.0f;
		VkDeviceQueueCreateInfo queueinfo{};
		queueinfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; //creates a structure to hold que family info
		queueinfo.queueFamilyIndex = indices.graphicsFamily.value(); // index of the queue family to create gotten from the findQueueFamilies function
		queueinfo.queueCount = 1;
		queueinfo.pQueuePriorities = &queuePriority;
		VkPhysicalDeviceFeatures deviceFeatures{}; //this struct is used to specify the features we will be using. such as geometry shaders, anisotropic filtering, etc.
		VkDeviceCreateInfo newinfo{}; //specify which queues to create
		newinfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		newinfo.pQueueCreateInfos = &queueinfo; //queues to create
		newinfo.queueCreateInfoCount = 1;
		newinfo.pEnabledFeatures = &deviceFeatures; //device features to enable
		const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME }; // specify the device extensions to enable
		newinfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		newinfo.ppEnabledExtensionNames = deviceExtensions.data();
		newinfo.enabledLayerCount = 0;
		newinfo.ppEnabledLayerNames = nullptr;
		if (vkCreateDevice(physicalDevice, &newinfo, nullptr, &device) != VK_SUCCESS) { // if logic device creation fails, output error
			throw std::runtime_error("failed to create logical device!");
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
	struct QueueFamilyIndices { // store the indices of the queue families that are supported by the device
		std::optional<uint32_t> graphicsFamily;

		bool isComplete() {
			return graphicsFamily.has_value();
		}
	};

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
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
			if (indices.isComplete()) {
				break;
			}
			i++;
		}
		return indices; //return the indices/position of the queue family that supports graphics
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);
		return indices.isComplete(); //checks if the quefamilies have all been searched and if the graphics family has been found
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
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
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
			throw std::runtime_error("failed to create swap chain!");
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
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { //if the format is srgb and the colorspace is srgb return the format, otherwise return the first format
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
			if (vkCreateImageView(device, &newinfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
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
	void setupGraphicsPipeline() {
		std::vector<char> vertShaderCode = readFile("vertex_shader.spv"); //read the vertex shader binary
		std::vector<char> fragShaderCode = readFile("fragment_shader.spv");
		vertShader = createShaderModule(vertShaderCode);
		fragShader = createShaderModule(fragShaderCode);
	}
	void createGraphicsPipeline(VkShaderModule vert, VkShaderModule frag) {
		// shader stage setup 
		VkPipelineShaderStageCreateInfo vertShader{}; //creates a struct for the vertex shader stage info
		vertShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShader.module = vert; //assign the vertex shader module
		vertShader.pName = "main";
		VkPipelineShaderStageCreateInfo fragShader{};
		fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShader.module = frag;
		fragShader.pName = "main";
		VkPipelineShaderStageCreateInfo stages[] = { vertShader, fragShader }; //create an array of the shader stage structs

		//vertex input setup 
		VkVertexInputBindingDescription bindDesc{}; //create a struct for the vertex input binding description
		bindDesc.binding = 0;
		bindDesc.stride = sizeof(Vertex); //num of bytes from 2 entrys
		bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; //the rate when data is loaded
		std::array<VkVertexInputAttributeDescription, 2> attrDesc; //attr0 is position, attr1 is color
		attrDesc[0].binding = 0;
		attrDesc[0].location = 0; // location in the vertex shader in vec3 position
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; //format is 3 32 bit floats for position
		attrDesc[0].offset = offsetof(Vertex, position);
		attrDesc[1].binding = 0;
		attrDesc[1].location = 1;
		attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attrDesc[1].offset = offsetof(Vertex, color);
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{}; //create a struct for the vertex input state
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;  //assign the struct type to the vertex input state
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindDesc;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
		vertexInputInfo.pVertexAttributeDescriptions = attrDesc.data(); //assign the vertex input attribute descriptions

		//input assembly setup (assembles the vertices into primitives)
		VkPipelineInputAssemblyStateCreateInfo inputAssem{}; //create a struct for the input assembly state
		inputAssem.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; //assign the struct type to the input assembly state
		inputAssem.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; //set the topology to triangle list (3 vertices per triangle)
		inputAssem.primitiveRestartEnable = VK_FALSE;

		//viewport and scissors setup (defines the region of the framebuffer that the output will be rendered to)
		VkViewport vp{}; //struct for the viewport
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

		//1. vertex input (done)
		//2. input assembly (done)
		//3. viewport and scissors (done)
		//4. rasterizer
		//5. multisampling
		//6. depth and stencil testing
		//7. color blending
		//8. pipeline layout
		//9. render pass
		//10. graphics pipeline


	}
	void initVulkan() {
		createInstance();
		pickDevice();
		createLogicalDevice();
		createSurface();
		createSC();
		setupGraphicsPipeline(); //reads the SPIRV binary and creates the shader modules
		createGraphicsPipeline(vertShader, fragShader); //takesn in the created shader modules and creates the graphics pipeline
	}
	void mainLoop() {
		while (!glfwWindowShouldClose(window)) { // while window is not closed
			glfwPollEvents(); // this function checks if any events are triggered
		}
	}
	void cleanup() {

		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
		vkDestroyShaderModule(device, vertShader, nullptr);
		vkDestroyShaderModule(device, fragShader, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}
	//TODO: 
	// 1. clean up code (done)
	// 2. set up the physical and logical devices. (done)
	// 3. create a swap chain to present images to the screen (done)
	// 4. create graphics pipeline to render the triangle (goal)
	// 5. create render passes, commandbuffers and framebuffers
	// 6. semaphores and fences for synchronization
	// 7. draw the triangle
};
int main() {
	Engine app; // 

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}