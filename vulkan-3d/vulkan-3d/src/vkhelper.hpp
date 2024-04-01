// A bunch of Vulkan helper functions for the project

#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <string>

extern VkDevice device;
extern VkQueue graphicsQueue;

class vkhelper {
public:
	static VkCommandBuffer beginSingleTimeCommands(const VkCommandPool cPool) {
		VkCommandBuffer commandBuffer = allocateCommandBuffers(cPool);
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; //one time command buffer
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}

	static void endSingleTimeCommands(const VkCommandBuffer cBuffer, const VkCommandPool cPool) {
		vkEndCommandBuffer(cBuffer);
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cBuffer;
		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE); //submit the command buffer to the queue
		vkQueueWaitIdle(graphicsQueue); //wait for the queue to be idle
		vkFreeCommandBuffers(device, cPool, 1, &cBuffer); //free the command buffer
	}

	static void createFB(const VkRenderPass renderPass, VkFramebuffer& frameBuf, const VkImageView& IV, const uint32_t width, const uint32_t height) {
		VkFramebufferCreateInfo frameBufferInfo{};
		frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferInfo.renderPass = renderPass;
		frameBufferInfo.attachmentCount = 1;
		frameBufferInfo.pAttachments = &IV; // imageview
		frameBufferInfo.width = width;
		frameBufferInfo.height = height;
		frameBufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &frameBuf) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer!");
		}
	}

	static void createFB(const VkRenderPass renderPass, VkFramebuffer& frameBuf, const std::vector<VkImageView>& attachments, const uint32_t width, const uint32_t height) {
		VkFramebufferCreateInfo frameBufferInfo{};
		frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferInfo.renderPass = renderPass;
		frameBufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		frameBufferInfo.pAttachments = attachments.data();
		frameBufferInfo.width = width;
		frameBufferInfo.height = height;
		frameBufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &frameBuf) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer!");
		}
	}

	static VkCommandBuffer allocateCommandBuffers(const VkCommandPool cPool, const VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
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

	static std::vector<VkCommandBuffer> allocateCommandBuffers(const VkCommandPool cPool, const size_t count, const VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
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
};