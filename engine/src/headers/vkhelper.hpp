// A bunch of Vulkan helper functions for the project

#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <optional>
#include <memory>

// ------------------ GLOBALS ------------------ //
extern VkDevice device;
extern VkQueue graphicsQueue;
extern VkPhysicalDevice physicalDevice;

// ------------------ FUNCTION POINTERS ------------------ //
namespace vkhfp {
#define FUNCTIONS \
    F(vkCreateAccelerationStructureKHR) \
    F(vkDestroyAccelerationStructureKHR) \
    F(vkGetAccelerationStructureBuildSizesKHR) \
    F(vkCmdBuildAccelerationStructuresKHR) \
    F(vkCmdWriteAccelerationStructuresPropertiesKHR) \
    F(vkCmdCopyAccelerationStructureKHR) \
    F(vkGetAccelerationStructureDeviceAddressKHR) \
    F(vkCmdPushDescriptorSetKHR)

#define F(name) inline PFN_##name name = nullptr;
	FUNCTIONS
#undef F

		template<typename T>
	void loadFunc(VkInstance instance, T& ptr, const char* name) {
		ptr = reinterpret_cast<T>(vkGetInstanceProcAddr(instance, name));
		if (!ptr) std::cerr << name << " isnt supported!" << std::endl;
	}

	inline void loadFuncPointers(VkInstance instance) {
#define F(name) loadFunc(instance, name, #name);
		FUNCTIONS
#undef F
	}
}

#define VKHFP_IMPLEMENTATION
// ------------------ RAII WRAPPERS ------------------ //
template<typename T, typename... Destroy>
struct VkhObject;

template<typename Object, typename... Destroy>
struct VkhObj {
	struct ObjWrapper {
		Object object;
		bool autoDestroy;
		std::tuple<Destroy...> destroyArgs;

		ObjWrapper(Object obj, bool destroy, Destroy... args) : object(obj), autoDestroy(destroy), destroyArgs(args...) {}

		// custom destructor
		~ObjWrapper() {
			if (autoDestroy && object != VK_NULL_HANDLE) {
				std::apply([this](auto&&... args) { VkhObject<Object, Destroy...>::destroy(object, args...); }, destroyArgs);
			}
		}

		// disallow moving and copying
		ObjWrapper(const ObjWrapper&) = delete;
		ObjWrapper& operator=(const ObjWrapper&) = delete;
		ObjWrapper(ObjWrapper&&) = delete;
		ObjWrapper& operator=(ObjWrapper&&) = delete;
	};

	std::shared_ptr<ObjWrapper> objectP;

	// constructors
	explicit VkhObj(Destroy... args) : objectP(std::make_shared<ObjWrapper>(VK_NULL_HANDLE, true, args...)) {}
	explicit VkhObj(Object obj, Destroy... args) : objectP(std::make_shared<ObjWrapper>(obj, true, args...)) {}

	// copy constructor and assignment
	VkhObj(const VkhObj& other) : objectP(other.objectP) {}
	VkhObj& operator=(const VkhObj& other) {
		if (this != &other) {
			objectP = other.objectP;
		}
		return *this;
	}

	// move constructor and assignment
	VkhObj(VkhObj&& other) noexcept = default;
	VkhObj& operator=(VkhObj&& other) noexcept = default;

	// destructor
	~VkhObj() = default;

	// equality operators
	bool operator==(const VkhObj& other) const noexcept {
		return objectP->object == other.objectP->object;
	}
	bool operator!=(const VkhObj& other) const noexcept { return !(*this == other); }

	bool valid() const noexcept { return objectP->object != VK_NULL_HANDLE; }

	explicit operator Object() const noexcept { return objectP->object; }

	// const
	const Object& v() const noexcept { return objectP->object; }
	const Object* p() const noexcept { return &objectP->object; }

	// not const
	Object& v() noexcept { return objectP->object; }
	Object* p() noexcept { return &objectP->object; }

	// get the current use count of the obj
	size_t use_count() const noexcept { return objectP.use_count(); }
	void setDestroy(bool destruction) { objectP->autoDestroy = destruction; }
};


template<typename T>
struct std::hash<VkhObj<T>> {
	size_t operator()(const VkhObj<T>& obj) const noexcept {
		return (obj.valid()) ? (std::hash<T>()(obj.v())) : 0;
	}
};

template<>
struct VkhObject<VkBuffer> {
	static void destroy(VkBuffer buffer) {
		std::cout << "buffer was destroyed: " << buffer << std::endl;
		vkDestroyBuffer(device, buffer, nullptr);
	}
};

template<>
struct VkhObject<VkDeviceMemory> {
	static void destroy(VkDeviceMemory mem) {
		std::cout << "memory was freed: " << mem << std::endl;
		vkFreeMemory(device, mem, nullptr);
	}
};

template<>
struct VkhObject<VkImage> {
	static void destroy(VkImage image) {
		std::cout << "image was freed: " << image << std::endl;
		vkDestroyImage(device, image, nullptr);
	}
};

template<>
struct VkhObject<VkImageView> {
	static void destroy(VkImageView imageView) {
		std::cout << "image view was freed: " << imageView << std::endl;
		vkDestroyImageView(device, imageView, nullptr);
	}
};

template<>
struct VkhObject<VkSampler> {
	static void destroy(VkSampler sampler) {
		std::cout << "sampler view was freed: " << sampler << std::endl;
		vkDestroySampler(device, sampler, nullptr);
	}
};

template<>
struct VkhObject<VkCommandPool> {
	static void destroy(VkCommandPool commandPool) {
		std::cout << "command pool was destroyed: " << commandPool << std::endl;
		vkDestroyCommandPool(device, commandPool, nullptr);
	}
};

template<>
struct VkhObject<VkCommandBuffer, VkCommandPool*> {
	static void destroy(VkCommandBuffer commandBuffer, VkCommandPool* commandPool) {
		std::cout << "command buffer was freed: " << commandBuffer << std::endl;
		vkFreeCommandBuffers(device, *commandPool, 1, &commandBuffer);
	}
};

template<>
struct VkhObject<VkDescriptorPool> {
	static void destroy(VkDescriptorPool descriptorPool) {
		std::cout << "descriptor pool was destroyed: " << descriptorPool << std::endl;
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
	}
};

template<>
struct VkhObject<VkDescriptorSetLayout> {
	static void destroy(VkDescriptorSetLayout descriptorSetLayout) {
		std::cout << "descriptor set layout was destroyed: " << descriptorSetLayout << std::endl;
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
	}
};

template<>
struct VkhObject<VkDescriptorSet, VkDescriptorPool*> {
	static void destroy(VkDescriptorSet descriptorSet, VkDescriptorPool* descriptorPool) {
		std::cout << "descriptor set was freed: " << descriptorSet << std::endl;
		vkFreeDescriptorSets(device, *descriptorPool, 1, &descriptorSet);
	}
};

template<>
struct VkhObject<VkPipeline> {
	static void destroy(VkPipeline pipeline) {
		std::cout << "pipeline was destroyed: " << pipeline << std::endl;
		vkDestroyPipeline(device, pipeline, nullptr);
	}
};

template<>
struct VkhObject<VkPipelineLayout> {
	static void destroy(VkPipelineLayout pipelineLayout) {
		std::cout << "pipeline layout was destroyed: " << pipelineLayout << std::endl;
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	}
};

template<>
struct VkhObject<VkShaderModule> {
	static void destroy(VkShaderModule shaderModule) {
		std::cout << "shader module was destroyed: " << shaderModule << std::endl;
		vkDestroyShaderModule(device, shaderModule, nullptr);
	}
};

template<>
struct VkhObject<VkRenderPass> {
	static void destroy(VkRenderPass renderpass) {
		std::cout << "renderpass was destroyed: " << renderpass << std::endl;
		vkDestroyRenderPass(device, renderpass, nullptr);
	}
};

template<>
struct VkhObject<VkFramebuffer> {
	static void destroy(VkFramebuffer framebuffer) {
		std::cout << "framebuffer was destroyed: " << framebuffer << std::endl;
		vkDestroyFramebuffer(device, framebuffer, nullptr);
	}
};

template<>
struct VkhObject<VkSemaphore> {
	static void destroy(VkSemaphore semaphore) {
		std::cout << "semaphore was destroyed: " << semaphore << std::endl;
		vkDestroySemaphore(device, semaphore, nullptr);
	}
};

template<>
struct VkhObject<VkFence> {
	static void destroy(VkFence fence) {
		std::cout << "fence was destroyed: " << fence << std::endl;
		vkDestroyFence(device, fence, nullptr);
	}
};

template<>
struct VkhObject<VkQueryPool> {
	static void destroy(VkQueryPool queryPool) {
		std::cout << "query pool was destroyed: " << queryPool << std::endl;
		vkDestroyQueryPool(device, queryPool, nullptr);
	}
};

using VkhBuffer = VkhObj<VkBuffer>;
using VkhDeviceMemory = VkhObj<VkDeviceMemory>;

using VkhImage = VkhObj<VkImage>;
using VkhImageView = VkhObj<VkImageView>;
using VkhSampler = VkhObj<VkSampler>;

using VkhCommandPool = VkhObj<VkCommandPool>;
using VkhCommandBuffer = VkhObj<VkCommandBuffer, VkCommandPool*>;

using VkhDescriptorPool = VkhObj<VkDescriptorPool>;
using VkhDescriptorSetLayout = VkhObj<VkDescriptorSetLayout>;
using VkhDescriptorSet = VkhObj<VkDescriptorSet, VkDescriptorPool*>;

using VkhPipeline = VkhObj<VkPipeline>;
using VkhPipelineLayout = VkhObj<VkPipelineLayout>;
using VkhShaderModule = VkhObj<VkShaderModule>;

using VkhRenderPass = VkhObj<VkRenderPass>;
using VkhFramebuffer = VkhObj<VkFramebuffer>;

using VkhSemaphore = VkhObj<VkSemaphore>;
using VkhFence = VkhObj<VkFence>;
using VkhQueryPool = VkhObj<VkQueryPool>;

class vkh {
public:
	typedef enum {
		BASE,
		DEPTH,
		NORMAL,
		METALLIC,
		EMISSIVE,
		OCCLUSION,
		CUBEMAP,
		ALPHA
	} TextureType;

	struct BufData {
		uint32_t vertexOffset;
		uint32_t vertexCount;
		uint32_t indexOffset;
		uint32_t indexCount;
	};

	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;
		std::optional<uint32_t> computeFamily;
		std::optional<uint32_t> transferFamily;

		bool graphicsComplete() {
			return graphicsFamily.has_value();
		}

		bool presentComplete() {
			return presentFamily.has_value();
		}

		bool computeComplete() {
			return computeFamily.has_value();
		}

		bool transferComplete() {
			return transferFamily.has_value();
		}

		bool allComplete() {
			return graphicsComplete() && presentComplete() && computeComplete() && transferComplete();
		}
	};

	struct SCsupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;

		SCsupportDetails()
			: capabilities(),
			formats(),
			presentModes()
		{}
	};

	// ------------------ SWAP CHAIN ------------------ //
	static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height) {
		VkExtent2D actualExtent = { width, height };

		// clamp the width and height to the min and max image extent
		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent; //return the actual extent
	}

	static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const VkSurfaceFormatKHR& format : availableFormats) {
			if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return format;
			}
		}

		return availableFormats[0];

	}
	static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const VkPresentModeKHR& present : availablePresentModes) {
			if (present == VK_PRESENT_MODE_MAILBOX_KHR) {
				return present;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	// gets the indices of each queue family (graphics, present, etc)
	static QueueFamilyIndices findQueueFamilyIndices(const VkSurfaceKHR& surface, const VkPhysicalDevice& device) {
		QueueFamilyIndices indices;

		// get the number of queue families and their properties
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		// iterate through all the queue families
		for (uint32_t i = 0; i < queueFamilyCount; i++) {
			const auto& family = queueFamilies[i];

			// check if the queue family supports graphics, compute and transfer operations
			if (family.queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphicsFamily = i;
			if (family.queueFlags & VK_QUEUE_COMPUTE_BIT) indices.computeFamily = i;
			if (family.queueFlags & VK_QUEUE_TRANSFER_BIT) indices.transferFamily = i;

			// check if the queue family supports presentation operations
			VkBool32 presSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presSupport);
			if (presSupport) indices.presentFamily = i;

			if (indices.allComplete()) {
				break;
			}
		}

		return indices;
	}

	// outputs details about what the swap chain supports
	static SCsupportDetails querySCsupport(const VkSurfaceKHR& surface) {
		SCsupportDetails details;

		// get the surface capabilities of the physical device
		// example: the minimum and maximum number of images in the swap chain
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

		// get the number of supported surface formats
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);

		// if the format count isnt 0, then get the actual surface format details
		// the surface format specifies the color space and pixel format
		if (formatCount) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
		}

		// get the number of supported present modes
		// present modes determine how the swapping of images to the display is handled
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);

		// if any present modes are supported, get the present mode details
		if (presentModeCount) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
		}
		return details;
	}

	// ------------------ MEMORY ------------------ //
	static uint32_t findMemoryType(const uint32_t tFilter, const VkMemoryPropertyFlags& prop) { //find the memory type based on the type filter and properties
		VkPhysicalDeviceMemoryProperties memP; //struct to hold memory properties
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memP); //get the memory properties for the physical device
		for (uint32_t i = 0; i < memP.memoryTypeCount; i++) { //loop through the memory types
			if ((tFilter & (1 << i)) && (memP.memoryTypes[i].propertyFlags & prop) == prop) { //if the memory type is suitable
				return i; //return the index of the memory type
			}
		}
		throw std::runtime_error("failed to find suitable memory type!");
	}

	static VkDeviceAddress bufferDeviceAddress(const VkhBuffer& buffer) {
		VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
		bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		bufferDeviceAddressInfo.buffer = buffer.v();

		return vkGetBufferDeviceAddress(device, &bufferDeviceAddressInfo);
	}

	static void createBuffer(VkhBuffer& buffer, VkhDeviceMemory& bufferMem, const VkDeviceSize& size, const VkBufferUsageFlags& usage, const VkMemoryPropertyFlags& memFlags, const VkMemoryAllocateFlags& memAllocFlags) {
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usage;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		// create the buffer
		if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, buffer.p()) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create buffer!");
		}

		// get the memory requirements for the buffer
		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(device, buffer.v(), &memoryRequirements);

		// mem allocation flags
		VkMemoryAllocateFlagsInfo allocFlagsInfo{};
		if (memAllocFlags) {
			allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
			allocFlagsInfo.flags = memAllocFlags;
		}

		// allocate mem for the buffer
		VkMemoryAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, memFlags);

		if (memAllocFlags) {
			allocateInfo.pNext = &allocFlagsInfo;
		}

		if (vkAllocateMemory(device, &allocateInfo, nullptr, bufferMem.p()) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate memory for the buffer!");
		}

		// bind the memory to the buffer
		if (vkBindBufferMemory(device, buffer.v(), bufferMem.v(), 0) != VK_SUCCESS) {
			throw std::runtime_error("Failed to bind memory to buffer!");
		}
	}

	static void createStagingBuffer(VkhBuffer& stagingBuffer, VkhDeviceMemory& stagingBufferMem, const VkDeviceSize& size, const VkMemoryAllocateFlags& memAllocFlags) {
		VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		createBuffer(stagingBuffer, stagingBufferMem, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, memFlags, memAllocFlags);
	}

	template<typename ObjType>
	static void createStagingBuffer(VkhBuffer& stagingBuffer, VkhDeviceMemory& stagingBufferMem, const ObjType& object, const VkDeviceSize& size, const VkMemoryAllocateFlags& memAllocFlags) {
		VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		createBuffer(stagingBuffer, stagingBufferMem, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, memFlags, memAllocFlags);

		// once memory is bound, map and fill it
		void* data;
		if (vkMapMemory(device, stagingBufferMem.v(), 0, size, 0, &data) != VK_SUCCESS) {
			throw std::runtime_error("Failed to map memory for buffer!");
		}

		// check if the object is trivally copyable
		if constexpr (std::is_trivially_copyable_v<ObjType>) {
			memcpy(data, object, size);
		}

		// if the object isnt trivially copyable
		else {
			memcpy(data, &object, size);
		}
		vkUnmapMemory(device, stagingBufferMem.v());
	}

	template<typename ObjType>
	static void createBuffer(VkhBuffer& buffer, VkhDeviceMemory& bufferMem, const ObjType& object, const VkDeviceSize& size, const VkBufferUsageFlags& usage,
		VkhCommandPool& commandPool, const VkQueue& queue, const VkMemoryAllocateFlags& memAllocFlags, bool staging = true) {
		createBuffer(buffer, bufferMem, size, usage, staging ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memAllocFlags);

		if (staging) {
			VkhBuffer stagingBuffer;
			VkhDeviceMemory stagingBufferMem;
			createStagingBuffer(stagingBuffer, stagingBufferMem, object, size, memAllocFlags);

			// copy the data from the staging buffer to the dst buffer
			VkhCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool);
			VkBufferCopy copyRegion{};
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer.v(), stagingBuffer.v(), buffer.v(), 1, &copyRegion);
			endSingleTimeCommands(commandBuffer, commandPool, queue);
		}
		else {
			void* data;
			if (vkMapMemory(device, bufferMem.v(), 0, size, 0, &data) != VK_SUCCESS) {
				throw std::runtime_error("Failed to map memory for buffer!");
			}

			// check if the object is trivally copyable
			if constexpr (std::is_trivially_copyable_v<ObjType>) {
				memcpy(data, object, size);
			}

			// if the object isnt trivially copyable
			else {
				memcpy(data, &object, size);
			}
			vkUnmapMemory(device, bufferMem.v());
		}
	}

	// ------------------ IMAGES ------------------ //
	static VkFormat findDepthFormat() {
		//the formats that are supported
		std::vector<VkFormat> allowed = {
			VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT
		};

		for (VkFormat format : allowed) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props); //get the format properties

			// if the format has the depth stencil attachment bit
			if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
				return format;
			}
		}
		throw std::runtime_error("failed to find suitable depth format!");
	}

	static void transitionImageLayout(const VkhCommandBuffer& commandBuffer, const VkhImage& image, const VkFormat format, const VkImageLayout oldLayout,
		const VkImageLayout newLayout, const uint32_t layerCount, const uint32_t levelCount, const uint32_t baseMip) {

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image.v();
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
		vkCmdPipelineBarrier(commandBuffer.v(), sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier); // insert the barrier into the command buffer
	}

	static void transitionImageLayout(VkhCommandPool& commandPool, const VkhImage& image, const VkFormat format, const VkImageLayout oldLayout, const VkImageLayout newLayout,
		const uint32_t layerCount, const uint32_t levelCount, const uint32_t baseMip) {
		VkhCommandBuffer tempCommandBuffer = beginSingleTimeCommands(commandPool);
		transitionImageLayout(tempCommandBuffer, image, format, oldLayout, newLayout, layerCount, levelCount, baseMip);
		endSingleTimeCommands(tempCommandBuffer, commandPool, graphicsQueue);
	}

	static void createImage(VkhImage& image, VkhDeviceMemory& imageMemory, const uint32_t width, const uint32_t height, const VkFormat format, const uint32_t mipLevels,
		const uint32_t arrayLayers, const bool cubeMap, const VkImageUsageFlags& usage, const VkSampleCountFlagBits& sample) {

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.arrayLayers = arrayLayers;
		imageInfo.mipLevels = mipLevels;
		imageInfo.format = format;

		// allows the gpu to format the image data in memory in the most efficient way
		// this means that the cpu cant easily read or write to the image though
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;

		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = sample;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (cubeMap) imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		if (vkCreateImage(device, &imageInfo, nullptr, image.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create color image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image.v(), &memRequirements);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;

		// memory is allocated on the gpu
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, imageMemory.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate color image memory!");
		}

		vkBindImageMemory(device, image.v(), imageMemory.v(), 0);
	}

	static void createImage(VkhImage& image, VkhDeviceMemory& imageMemory, const uint32_t width, const uint32_t height, const VkFormat format, const uint32_t mipLevels,
		const uint32_t arrayLayers, const bool cubeMap, const VkImageUsageFlags& usage, const VkImageLayout& imageLayout, VkhCommandPool& commandPool, const VkSampleCountFlagBits& sample) {

		createImage(image, imageMemory, width, height, format, mipLevels, arrayLayers, cubeMap, usage, sample);
		transitionImageLayout(commandPool, image, format, VK_IMAGE_LAYOUT_UNDEFINED, imageLayout, arrayLayers, mipLevels, 0);
	}

	static void createSampler(VkhSampler& sampler, const uint32_t mipLevels, const TextureType type = BASE) {
		VkSamplerCreateInfo samplerInf{};
		samplerInf.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInf.magFilter = VK_FILTER_LINEAR; // magnification filter
		samplerInf.minFilter = VK_FILTER_LINEAR; // minification filter

		// when the texture cords go out of bounds, repeat the texture
		samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInf.anisotropyEnable = VK_FALSE;
		samplerInf.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInf.unnormalizedCoordinates = VK_FALSE;
		switch (type) {
		case DEPTH:
			// when the texture cords go out of bounds, clamp the uv cords to the edge of the texture
			samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

			// instead of directly returning a sampled image, the sampler will compare a refrence value to the sampled value
			// this is particularly useful for shadowmapping
			samplerInf.compareEnable = VK_TRUE;
			samplerInf.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
			break;
		case CUBEMAP:
			samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			break;
		}
		samplerInf.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInf.minLod = 0.0f;
		samplerInf.maxLod = static_cast<float>(mipLevels);
		if (vkCreateSampler(device, &samplerInf, nullptr, sampler.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	template<typename Texture>
	static void createImageView(Texture& tex, const TextureType type = BASE) {
		VkImageViewCreateInfo viewInf{};
		viewInf.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInf.image = tex.image.v();
		viewInf.subresourceRange.baseArrayLayer = 0;
		viewInf.subresourceRange.layerCount = 1;
		viewInf.viewType = VK_IMAGE_VIEW_TYPE_2D;

		switch (type) {
		case BASE:
			viewInf.format = VK_FORMAT_R8G8B8A8_SRGB;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			break;
		case DEPTH:
			viewInf.format = VK_FORMAT_D32_SFLOAT;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			break;
		case NORMAL:
			viewInf.format = VK_FORMAT_R8G8B8A8_UNORM;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			break;
		case METALLIC:
			viewInf.format = VK_FORMAT_R8G8B8A8_UNORM;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			break;
		case EMISSIVE:
			viewInf.format = VK_FORMAT_R8G8B8A8_SRGB;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			break;
		case OCCLUSION:
			viewInf.format = VK_FORMAT_R8G8B8A8_UNORM;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			break;
		case CUBEMAP:
			viewInf.format = VK_FORMAT_R32G32B32A32_SFLOAT;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInf.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
			viewInf.subresourceRange.layerCount = 6;
			break;
		case ALPHA:
			viewInf.format = VK_FORMAT_R32_SFLOAT;
			viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			break;
		default:
			throw std::invalid_argument("Invalid texture type!");
		}

		viewInf.subresourceRange.baseMipLevel = 0;
		uint32_t level = (tex.mipLevels <= 0) ? 1 : tex.mipLevels;
		viewInf.subresourceRange.levelCount = level - viewInf.subresourceRange.baseMipLevel;
		if (vkCreateImageView(device, &viewInf, nullptr, tex.imageView.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}
	}

	template<typename Texture>
	static void createImageView(Texture& tex, const VkFormat& swapFormat) { // imageview creation for swapchain image types
		VkImageViewCreateInfo viewInf{};
		viewInf.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInf.image = tex.image.v();
		viewInf.subresourceRange.baseArrayLayer = 0;
		viewInf.subresourceRange.layerCount = 1;
		viewInf.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInf.format = swapFormat;
		viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInf.subresourceRange.baseMipLevel = 0;

		uint32_t level = (tex.mipLevels <= 0) ? 1 : tex.mipLevels;
		viewInf.subresourceRange.levelCount = level - viewInf.subresourceRange.baseMipLevel;
		if (vkCreateImageView(device, &viewInf, nullptr, tex.imageView.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view! (swap)");
		}
	}

	// copy an image from one image to another
	static void copyImage(VkhImage& srcImage, VkhImage& dstImage, const VkImageLayout& srcStart, const VkImageLayout dstStart, const VkImageLayout dstAfter, const VkhCommandBuffer& commandBuffer, const VkFormat format, const uint32_t width, const uint32_t height, const bool color) {
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

		vkCmdCopyImage(commandBuffer.v(), srcImage.v(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage.v(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
		transitionImageLayout(commandBuffer, dstImage, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, dstAfter, 1, 1, 0);
		transitionImageLayout(commandBuffer, srcImage, format, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1, 0);
	}

	static void copyImage(VkhCommandPool& commandPool, VkhImage& srcImage, VkhImage& dstImage, const VkImageLayout srcStart, const VkImageLayout dstStart, const VkImageLayout dstAfter, const VkFormat format, const uint32_t width, const uint32_t height, const bool color) {
		VkhCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool);
		copyImage(srcImage, dstImage, srcStart, dstStart, dstAfter, commandBuffer, format, width, height, color);
		endSingleTimeCommands(commandBuffer, commandPool, graphicsQueue);
	}


	// ------------------ COMMAND BUFFERS ------------------ //
	static VkhCommandPool createCommandPool(const uint32_t queueFamilyIndex, const VkCommandPoolCreateFlags& createFlags) {
		VkhCommandPool commandPool;
		VkCommandPoolCreateInfo poolInf{};
		poolInf.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInf.queueFamilyIndex = queueFamilyIndex; // the queue family that will be using this command pool
		poolInf.flags = createFlags;
		VkResult result = vkCreateCommandPool(device, &poolInf, nullptr, commandPool.p());
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
		return commandPool;
	}

	static VkhCommandBuffer beginSingleTimeCommands(VkhCommandPool& commandPool) {
		VkhCommandBuffer commandBuffer = allocateCommandBuffers(commandPool);
		commandBuffer.setDestroy(false); // command buffer wont be autodestroyed
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; //one time command buffer
		vkBeginCommandBuffer(commandBuffer.v(), &beginInfo);
		return commandBuffer;
	}

	static void endSingleTimeCommands(VkhCommandBuffer& commandBuffer, const VkhCommandPool& commandPool, const VkQueue& queue) {
		vkEndCommandBuffer(commandBuffer.v());
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = commandBuffer.p();
		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE); //submit the command buffer to the queue
		vkQueueWaitIdle(queue); //wait for the queue to be idle
		vkFreeCommandBuffers(device, commandPool.v(), 1, commandBuffer.p());
	}

	static void createFB(const VkhRenderPass& renderPass, VkhFramebuffer& frameBuf, const VkImageView* attachments, const size_t attatchmentCount, const uint32_t width, const uint32_t height) {
		VkFramebufferCreateInfo frameBufferInfo{};
		frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferInfo.renderPass = renderPass.v();
		frameBufferInfo.attachmentCount = static_cast<uint32_t>(attatchmentCount);
		frameBufferInfo.pAttachments = attachments;
		frameBufferInfo.width = width;
		frameBufferInfo.height = height;
		frameBufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &frameBufferInfo, nullptr, frameBuf.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer!");
		}
	}

	static VkhCommandBuffer allocateCommandBuffers(VkhCommandPool& commandPool, const VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
		VkhCommandBuffer commandBuffer(commandPool.p());
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool.v();
		allocInfo.level = level;
		allocInfo.commandBufferCount = 1;

		VkResult result = vkAllocateCommandBuffers(device, &allocInfo, commandBuffer.p());
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffer!!");
		}
		return commandBuffer;
	}

	static void createSemaphore(VkhSemaphore& semaphore) {
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkResult resultRenderFinished = vkCreateSemaphore(device, &semaphoreInfo, nullptr, semaphore.p());
		if (resultRenderFinished != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphore!");
		}
	}

	static VkSubmitInfo createSubmitInfo(const VkhCommandBuffer* commandBuffers, const size_t commandBufferCount) {
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = static_cast<uint32_t>(commandBufferCount);
		submitInfo.pCommandBuffers = commandBuffers->p();
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;
		return submitInfo;
	}

	static VkSubmitInfo createSubmitInfo(const VkhCommandBuffer* commandBuffers, const size_t commandBufferCount, const VkPipelineStageFlags* waitStages, const VkhSemaphore& wait, const VkhSemaphore& signal) {
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = wait.p();
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = static_cast<uint32_t>(commandBufferCount);
		submitInfo.pCommandBuffers = commandBuffers->p();
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signal.p();
		return submitInfo;
	}

	static VkSubmitInfo createSubmitInfo(const VkhCommandBuffer* commandBuffers, const VkPipelineStageFlags* waitStages, const VkhSemaphore* wait, const VkhSemaphore* signal,
		const size_t commandBufferCount, const size_t waitSemaphoreCount, const size_t signalSemaphoreCount) {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphoreCount);
		submitInfo.pWaitSemaphores = wait->p();
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = static_cast<uint32_t>(commandBufferCount);
		submitInfo.pCommandBuffers = commandBuffers->p();
		submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphoreCount);
		submitInfo.pSignalSemaphores = signal->p();
		return submitInfo;
	}

	// ------------------ DESCRIPTOR SETS ------------------ //
	static VkhDescriptorSetLayout createDSLayout(const uint32_t bindingIndex, const VkDescriptorType& type, const uint32_t descriptorCount, const VkShaderStageFlags& stageFlags, const bool pushDescriptors = false) {
		VkDescriptorSetLayoutBinding binding{};
		binding.binding = bindingIndex;
		binding.descriptorType = type;
		binding.descriptorCount = descriptorCount;
		binding.stageFlags = stageFlags;

		// if descriptorCount is over 1, set the binding flag to indicate a variable descriptor count
		// this is used when the number of things sent to the shader isnt known when the pipeline is created
		VkDescriptorBindingFlagsEXT bindingFlags = 0;
		if (descriptorCount > 1) {
			bindingFlags |= VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT;
		}

		VkDescriptorSetLayoutBindingFlagsCreateInfoEXT bindingFlagsInfo{};
		bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
		bindingFlagsInfo.bindingCount = 1;
		bindingFlagsInfo.pBindingFlags = &bindingFlags;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.pNext = &bindingFlagsInfo;
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &binding;
		if (pushDescriptors) layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

		VkhDescriptorSetLayout descriptorSetLayout;
		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout.p()) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor set layout!");
		}

		return descriptorSetLayout;
	}

	static VkhDescriptorPool createDSPool(const VkDescriptorType& type, const uint32_t descriptorCount) {
		VkDescriptorPoolSize poolSize{};
		poolSize.type = type;
		poolSize.descriptorCount = descriptorCount;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		poolInfo.maxSets = 1;

		VkhDescriptorPool descriptorPool;
		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, descriptorPool.p()) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor pool!");
		}

		return descriptorPool;
	}

	template<typename InfoType>
	static VkWriteDescriptorSet createDSWrite(const VkhDescriptorSet& set, const uint32_t binding, const uint32_t arrayElem, const VkDescriptorType& type, const InfoType* infos, const size_t count) {
		static_assert(std::is_same_v<InfoType, VkDescriptorImageInfo> || std::is_same_v<InfoType, VkDescriptorBufferInfo>, "Invalid info type");

		VkWriteDescriptorSet d{};
		d.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		d.dstSet = set.v();
		d.dstBinding = binding;
		d.dstArrayElement = arrayElem;
		d.descriptorType = type;
		d.descriptorCount = static_cast<uint32_t>(count);

		if constexpr (std::is_same_v<InfoType, VkDescriptorImageInfo>) { // if the info type is an image
			d.pImageInfo = infos;
		}
		else if constexpr (std::is_same_v<InfoType, VkDescriptorBufferInfo>) { // if the info type is a buffer
			d.pBufferInfo = infos;
		}

		return d;
	}

	template<typename InfoType>
	static VkWriteDescriptorSet createDSWrite(const VkhDescriptorSet& set, const uint32_t binding, const uint32_t arrayElem, const VkDescriptorType& type, const InfoType& info) {
		static_assert(std::is_same_v<InfoType, VkDescriptorImageInfo> || std::is_same_v<InfoType, VkDescriptorBufferInfo>, "Invalid info type");

		VkWriteDescriptorSet d{};
		d.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		d.dstSet = set.v();
		d.dstBinding = binding;
		d.dstArrayElement = arrayElem;
		d.descriptorType = type;
		d.descriptorCount = 1;

		if constexpr (std::is_same_v<InfoType, VkDescriptorImageInfo>) { // if the info type is an image
			d.pImageInfo = &info;
		}
		else if constexpr (std::is_same_v<InfoType, VkDescriptorBufferInfo>) { // if the info type is a buffer
			d.pBufferInfo = &info;
		}

		return d;
	}

	// ------------------ PIPELINES ------------------ //
	static VkhShaderModule createShaderModule(const std::vector<char>& code) { //takes in SPIRV binary and creates a shader module
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); //convert the char array to uint32_t array

		VkhShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, shaderModule.p()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}
};