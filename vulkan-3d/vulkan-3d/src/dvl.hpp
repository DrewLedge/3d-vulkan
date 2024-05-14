// Drew's Vertex Library (DVL)
// Designed to work with Vulkan and GLTF 2.0

#include <vulkan/vulkan.h>
#pragma once

class dvl {
public:
	struct Vertex {
		dml::vec3 pos; // position coordinates x, y, z
		dml::vec2 tex; // texture coordinates u, v
		dml::vec4 col; // color r, g, b, a
		dml::vec3 normal; // normal vector x, y, z
		dml::vec4 tangent;

		// default constructor:
		Vertex()
			: pos(dml::vec3(0.0f, 0.0f, 0.0f)),
			tex(dml::vec2(0.0f, 0.0f)),
			col(dml::vec4(0.0f, 0.0f, 0.0f, 0.0f)),
			normal(dml::vec3(0.0f, 0.0f, 0.0f)),
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
			tangent(tang)
		{}
		bool operator==(const Vertex& other) const {
			const float epsilon = 0.00001f; // tolerance for floating point equality
			return pos == other.pos &&
				tex == other.tex &&
				col == other.col &&
				normal == other.normal &&
				tangent == other.tangent;
		}
	};

	struct VertHash {
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

	struct TexHash {
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

	struct Material {
		Texture metallicRoughness;
		Texture baseColor;
		Texture normalMap;
		Texture occlusionMap;
		Texture emissiveMap;
	};

	struct Mesh {
		Material material; //used to store all the textures/materials of the mesh
		std::vector<dvl::Vertex> vertices;
		std::vector<uint32_t> indices;
		std::string pathObj; // i.e "models/cube.obj"

		dml::vec3 position;  // position of the mesh
		dml::vec4 rotation;  // rotation of the mesh in quaternions
		dml::vec3 scale;     // scale of the mesh
		dml::mat4 modelMatrix;

		size_t textureCount; // number of textures in the mesh
		size_t texIndex; // where in the texture array the textures of the mesh start

		bool startObj; // wether is loaded at the start of the program or not
		bool player; // if the object is treated as a player mesh or not

		size_t meshHash;
		std::string name;

		// default constructor
		Mesh()
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
			meshHash(),
			name("")
		{}

		// copy constructor
		Mesh(const Mesh& other)
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
			meshHash(other.meshHash),
			name(other.name) {
		}
	};

	template <typename IndexType>
	static void calculateTangents(const float* positionData, const float* texCoordData, std::vector<dml::vec4>& tangents,
		const void* rawIndices, size_t size) {

		for (size_t i = 0; i < size; i += 3) {
			IndexType i1 = static_cast<const IndexType*>(rawIndices)[i];
			IndexType i2 = static_cast<const IndexType*>(rawIndices)[i + 1];
			IndexType i3 = static_cast<const IndexType*>(rawIndices)[i + 2];

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
	}

	static void normalizeTangents(std::vector<dml::vec4>& tangents) {
		for (dml::vec4& tangent : tangents) {
			dml::vec3 normalizedTangent = dml::normalize(tangent.xyz());
			tangent.x = normalizedTangent.x;
			tangent.y = normalizedTangent.y;
			tangent.z = normalizedTangent.z;
		}
	}
};
