// A bunch of Vulkan helper functions for the project

#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <string>

extern VkDevice device;
extern VkQueue graphicsQueue;
extern VkPhysicalDevice physicalDevice;

class vkhelper {
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

	static void createBuffer(VkBuffer& buffer, VkDeviceMemory& bufferMem, const VkDeviceSize& size, const VkBufferUsageFlags& usage) {
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usage;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		// create the buffer
		if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create buffer!");
		}

		// get the memory requirements for the buffer
		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

		// allocate mem for the buffer
		VkMemoryAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		if (vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMem) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate memory for the buffer!");
		}

		// bind the memory to the buffer
		if (vkBindBufferMemory(device, buffer, bufferMem, 0) != VK_SUCCESS) {
			throw std::runtime_error("Failed to bind memory to buffer!");
		}
	}

	template<typename ObjType>
	static void createBuffer(VkBuffer& buffer, VkDeviceMemory& bufferMem, const ObjType& object, const VkDeviceSize& size, const VkBufferUsageFlags& usage) {
		createBuffer(buffer, bufferMem, size, usage);

		// once memory is bound, map and fill it
		void* data;
		if (vkMapMemory(device, bufferMem, 0, size, 0, &data) != VK_SUCCESS) {
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
		vkUnmapMemory(device, bufferMem);
	}

	// ------------------ IMAGES ------------------ //
	static void transitionImageLayout(const VkCommandBuffer& commandBuffer, const VkImage& image, const VkFormat format, const VkImageLayout oldLayout,
		const VkImageLayout newLayout, const uint32_t layerCount, const uint32_t levelCount, const uint32_t baseMip) {

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

	static void transitionImageLayout(const VkCommandPool& cPool, const VkImage& image, const VkFormat format, const VkImageLayout oldLayout, const VkImageLayout newLayout,
		const uint32_t layerCount, const uint32_t levelCount, const uint32_t baseMip) {
		VkCommandBuffer tempCommandBuffer = beginSingleTimeCommands(cPool);
		transitionImageLayout(tempCommandBuffer, image, format, oldLayout, newLayout, layerCount, levelCount, baseMip);
		endSingleTimeCommands(tempCommandBuffer, cPool, graphicsQueue);
	}

	static void createImage(VkImage& image, VkDeviceMemory& imageMemory, const uint32_t width, const uint32_t height, const VkFormat format, const uint32_t mipLevels,
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
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = sample;
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

	static void createImage(VkImage& image, VkDeviceMemory& imageMemory, const uint32_t width, const uint32_t height, const VkFormat format, const uint32_t mipLevels,
		const uint32_t arrayLayers, const bool cubeMap, const VkImageUsageFlags& usage, const VkImageLayout& imageLayout, const VkCommandPool& cPool, const VkSampleCountFlagBits& sample) {

		createImage(image, imageMemory, width, height, format, mipLevels, arrayLayers, cubeMap, usage, sample);
		transitionImageLayout(cPool, image, format, VK_IMAGE_LAYOUT_UNDEFINED, imageLayout, arrayLayers, mipLevels, 0);
	}

	static void createSampler(VkSampler& sampler, const uint32_t mipLevels, const TextureType type = BASE) {
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
		switch (type) {
		case DEPTH:
			samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.compareEnable = VK_TRUE;
			samplerInf.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
			samplerInf.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
			break;
		case CUBEMAP:
			samplerInf.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInf.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // prevent seams at the edges
			break;
		}
		samplerInf.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInf.minLod = 0.0f;
		samplerInf.maxLod = static_cast<float>(mipLevels);
		if (vkCreateSampler(device, &samplerInf, nullptr, &sampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	template<typename Texture>
	static void createImageView(Texture& tex, const TextureType type = BASE) {
		VkImageViewCreateInfo viewInf{};
		viewInf.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInf.image = tex.image;
		viewInf.subresourceRange.baseArrayLayer = 0;
		viewInf.subresourceRange.layerCount = 1;
		viewInf.viewType = VK_IMAGE_VIEW_TYPE_2D;

		switch (type) {
		case BASE:
			viewInf.format = VK_FORMAT_R8G8B8A8_SRGB; // for base texture
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
			viewInf.format = VK_FORMAT_R8G8B8A8_UNORM; // for metallic roughness
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
			viewInf.format = VK_FORMAT_R32G32B32A32_SFLOAT; // for cubemaps
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
		if (vkCreateImageView(device, &viewInf, nullptr, &tex.imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}
	}

	template<typename Texture>
	static void createImageView(Texture& tex, const VkFormat& swapFormat) { // imageview creation for swapchain image types
		VkImageViewCreateInfo viewInf{};
		viewInf.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInf.image = tex.image;
		viewInf.subresourceRange.baseArrayLayer = 0;
		viewInf.subresourceRange.layerCount = 1;
		viewInf.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInf.format = swapFormat; // format of the swap chain
		viewInf.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInf.subresourceRange.baseMipLevel = 0;

		uint32_t level = (tex.mipLevels <= 0) ? 1 : tex.mipLevels;
		viewInf.subresourceRange.levelCount = level - viewInf.subresourceRange.baseMipLevel;
		if (vkCreateImageView(device, &viewInf, nullptr, &tex.imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view! (swap)");
		}
	}

	// copy an image from one image to another
	static void copyImage(VkImage& srcImage, VkImage& dstImage, const VkImageLayout& srcStart, const VkImageLayout dstStart, const VkImageLayout dstAfter, const VkCommandBuffer& commandBuffer, const VkFormat format, const uint32_t width, const uint32_t height, const bool color) {
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

	static void copyImage(const VkCommandPool& cPool, VkImage& srcImage, VkImage& dstImage, const VkImageLayout srcStart, const VkImageLayout dstStart, const VkImageLayout dstAfter, const VkFormat format, const uint32_t width, const uint32_t height, const bool color) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands(cPool);
		copyImage(srcImage, dstImage, srcStart, dstStart, dstAfter, commandBuffer, format, width, height, color);
		endSingleTimeCommands(commandBuffer, cPool, graphicsQueue);
	}


	// ------------------ COMMAND BUFFERS ------------------ //
	static VkCommandBuffer beginSingleTimeCommands(const VkCommandPool& cPool) {
		VkCommandBuffer commandBuffer = allocateCommandBuffers(cPool);
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; //one time command buffer
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}

	static void endSingleTimeCommands(const VkCommandBuffer& cBuffer, const VkCommandPool& cPool, const VkQueue& queue) {
		vkEndCommandBuffer(cBuffer);
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cBuffer;
		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE); //submit the command buffer to the queue
		vkQueueWaitIdle(queue); //wait for the queue to be idle
		vkFreeCommandBuffers(device, cPool, 1, &cBuffer); //free the command buffer
	}

	static void createFB(const VkRenderPass& renderPass, VkFramebuffer& frameBuf, const VkImageView* attachments, const size_t attatchmentCount, const uint32_t width, const uint32_t height) {
		if (frameBuf != VK_NULL_HANDLE) vkDestroyFramebuffer(device, frameBuf, nullptr);

		VkFramebufferCreateInfo frameBufferInfo{};
		frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferInfo.renderPass = renderPass;
		frameBufferInfo.attachmentCount = static_cast<uint32_t>(attatchmentCount);
		frameBufferInfo.pAttachments = attachments;
		frameBufferInfo.width = width;
		frameBufferInfo.height = height;
		frameBufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &frameBuf) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer!");
		}
	}

	static VkCommandBuffer allocateCommandBuffers(const VkCommandPool& cPool, const VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
		VkCommandBuffer commandBuffer;
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = cPool;
		allocInfo.level = level;
		allocInfo.commandBufferCount = 1;

		VkResult result = vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffer!!");
		}
		return commandBuffer;
	}

	static std::vector<VkCommandBuffer> allocateCommandBuffers(const VkCommandPool& cPool, const size_t count, const VkCommandBufferLevel& level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
		std::vector<VkCommandBuffer> commandBuffers(count);
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = cPool;
		allocInfo.level = level;
		allocInfo.commandBufferCount = static_cast<uint32_t>(count);

		VkResult result = vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!!");
		}
		return commandBuffers;
	}

	static void createSemaphore(VkSemaphore& semaphore) {
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkResult resultRenderFinished = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore);
		if (resultRenderFinished != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphore!");
		}
	}

	static VkSubmitInfo createSubmitInfo(const VkCommandBuffer* commandBuffers, const size_t commandBufferCount) {
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = static_cast<uint32_t>(commandBufferCount);
		submitInfo.pCommandBuffers = commandBuffers;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;
		return submitInfo;
	}

	static VkSubmitInfo createSubmitInfo(const VkCommandBuffer* commandBuffers, const size_t commandBufferCount, const VkPipelineStageFlags* waitStages, const VkSemaphore& wait, const VkSemaphore& signal) {
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &wait;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = static_cast<uint32_t>(commandBufferCount);
		submitInfo.pCommandBuffers = commandBuffers;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &signal;
		return submitInfo;
	}

	static VkSubmitInfo createSubmitInfo(const VkCommandBuffer* commandBuffers, const VkPipelineStageFlags* waitStages, const VkSemaphore* wait, const VkSemaphore* signal,
		const size_t commandBufferCount, const size_t waitSemaphoreCount, const size_t signalSemaphoreCount) {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphoreCount);
		submitInfo.pWaitSemaphores = wait;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = static_cast<uint32_t>(commandBufferCount);
		submitInfo.pCommandBuffers = commandBuffers;
		submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphoreCount);
		submitInfo.pSignalSemaphores = signal;
		return submitInfo;
	}

	// ------------------ DESCRIPTOR SETS ------------------ //
	static VkDescriptorSetLayout createDSLayout(const uint32_t bindingIndex, const VkDescriptorType& type, const uint32_t descriptorCount, const VkShaderStageFlags& stageFlags, const bool pushDescriptors = false) {
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
		if (pushDescriptors) layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

		VkDescriptorSetLayout descriptorSetLayout;
		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor set layout!");
		}

		return descriptorSetLayout;
	}

	static VkDescriptorPool createDSPool(const VkDescriptorType& type, const uint32_t descriptorCount) {
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

	template<typename InfoType>
	static VkWriteDescriptorSet createDSWrite(const VkDescriptorSet& set, const uint32_t binding, const uint32_t arrayElem, const VkDescriptorType& type, const InfoType* infos, const size_t count) {
		static_assert(std::is_same_v<InfoType, VkDescriptorImageInfo> || std::is_same_v<InfoType, VkDescriptorBufferInfo>, "Invalid info type");

		VkWriteDescriptorSet d{};
		d.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		d.dstSet = set;
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
	static VkWriteDescriptorSet createDSWrite(const VkDescriptorSet& set, const uint32_t binding, const uint32_t arrayElem, const VkDescriptorType& type, const InfoType& info) {
		static_assert(std::is_same_v<InfoType, VkDescriptorImageInfo> || std::is_same_v<InfoType, VkDescriptorBufferInfo>, "Invalid info type");

		VkWriteDescriptorSet d{};
		d.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		d.dstSet = set;
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
};