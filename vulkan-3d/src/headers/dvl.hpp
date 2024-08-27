// Drew's Vertex Library (DVL)
// Designed to work with Vulkan and GLTF 2.0

#pragma once

#include <vulkan/vulkan.h>
#include <unordered_set>
#include <dml.hpp>
#include <utils.hpp>

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
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;

		dml::vec3 position;
		dml::vec4 rotation;
		dml::vec3 scale;
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

	// returns an iterator to the attribute from a given name (TEXCOORD_0, NORMAL, etc)
	static auto getAttributeIt(const std::string& name, const std::map < std::string, int>& attributes) {
		auto it = attributes.find(name);

		// if the attribute isnt found, log a warning (if debug is enabled)
		LOG_WARNING_IF("Failed to find attribute: " + name, it == attributes.end());
		return it;
	}

	// returns a pointer to the beggining of the attribute data
	static const float* getAccessorData(const tinygltf::Model& model, const std::map<std::string, int>& attributes, const std::string& attributeName) {
		auto it = getAttributeIt(attributeName, attributes); // get the attribute iterator from the attribute name
		if (it == attributes.end()) return nullptr; // if the attribute isnt found, return nullptr

		// get the accessor object from the attribite name
		// accessors are objects that describe how to access the binary data in the tinygltf model as meaningful data
		// it->second is the index of the accessor in the models accessors array
		const tinygltf::Accessor& accessor = model.accessors[it->second];

		// get the buffer view from the accessor
		// the bufferview describes data about the buffer (stride, length, offset, etc)
		// accessor.bufferview is the index of the bufferview to use
		const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];

		// get the buffer based from the buffer view
		// the buffer is the raw binary data of the model
		// bufferView.buffer is the index of the buffer to use
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

		// get the offset of the accessor in the buffer
		// bufferView.byteOffset is the offset of the buffer view inside the buffer
		// accessor.byteOffset is the offset of the accessor in the buffer view
		// the sum gives the total offset from the start to the beginning of the attribute data
		size_t offset = bufferView.byteOffset + accessor.byteOffset;


		// return the data from the buffer marking the start of the attribute data!
		return reinterpret_cast<const float*>(&buffer.data[offset]);
	}

	// returns a pointer to the start of the index data (indices of the mesh)
	static const void* getIndexData(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
		// get the buffer view and buffer
		const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

		// get the offset of the accessor in the buffer
		size_t offset = bufferView.byteOffset + accessor.byteOffset;

		// go through the accessors component type
		// the compoenent type is the datatype of the data thats being read
		// from this data, cast the binary data (of the buffer) to the correct type
		switch (accessor.componentType) {
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
			return reinterpret_cast<const uint8_t*>(&buffer.data[offset]);
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
			return reinterpret_cast<const uint16_t*>(&buffer.data[offset]);
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
			return reinterpret_cast<const uint32_t*>(&buffer.data[offset]);
		default:
			// if the component type is not supported, log a warning and return nullptr
			LOG_WARNING("Unsupported index type" + accessor.componentType);
			return nullptr;
		}
	}

	static dml::mat4 calcNodeLM(const tinygltf::Node& node) { // get the local matrix of the node
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

	static int getNodeIndex(const tinygltf::Model& model, int meshIndex) {
		for (size_t i = 0; i < model.nodes.size(); i++) {
			if (model.nodes[i].mesh == meshIndex) {
				return static_cast<int>(i);
			}
		}
		return -1; // not found
	}

	static dml::mat4 calcMeshWM(const tinygltf::Model& gltfMod, int meshIndex, std::unordered_map<int, int>& parentIndex, Mesh& m) {
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

	static void printNodeHierarchy(const tinygltf::Model& model, int nodeIndex, int depth = 0) {
		for (int i = 0; i < depth; i++) { // indent based on depth
			std::cout << "     ";
		}
		// print the current node's name or index if the name is empty
		std::cout << "Node: " << (model.nodes[nodeIndex].name.empty() ? std::to_string(nodeIndex) : model.nodes[nodeIndex].name) << std::endl;

		for (const int& childIndex : model.nodes[nodeIndex].children) {
			printNodeHierarchy(model, childIndex, depth + 1);
		}
	}

	static void printFullHierarchy(const tinygltf::Model& model) {
		std::unordered_set<int> childNodes;
		for (const tinygltf::Node& node : model.nodes) {
			for (const int& childIndex : node.children) {
				childNodes.insert(childIndex);
			}
		}

		for (int i = 0; i < model.nodes.size(); i++) {
			if (childNodes.find(i) == childNodes.end()) { // if a node doesn't appear in the childNodes set, it's a root
				printNodeHierarchy(model, i);
				utils::sep();
			}
		}
	}

	static Mesh loadMesh(const tinygltf::Mesh& mesh, tinygltf::Model& model, std::unordered_map<int, int>& parentInd,
		const uint32_t meshInd, const dml::vec3 scale, const dml::vec3 pos, const dml::vec4 rot) {

		Mesh newObject;

		std::unordered_map<Vertex, uint32_t, VertHash> uniqueVertices;
		std::vector<Vertex> tempVertices;
		std::vector<uint32_t> tempIndices;

		//printFullHierarchy(model);

		// process primitives in the mesh
		for (const tinygltf::Primitive& primitive : mesh.primitives) {
			LOG_WARNING_IF("Unsupported primitive mode: " + std::to_string(primitive.mode), primitive.mode != TINYGLTF_MODE_TRIANGLES);

			const float* positionData = getAccessorData(model, primitive.attributes, "POSITION");
			const float* texCoordData = getAccessorData(model, primitive.attributes, "TEXCOORD_0");
			const float* normalData = getAccessorData(model, primitive.attributes, "NORMAL");
			const float* colorData = getAccessorData(model, primitive.attributes, "COLOR_0");
			const float* tangentData = getAccessorData(model, primitive.attributes, "TANGENT");

			if (!positionData || !texCoordData || !normalData) {
				throw std::runtime_error("Mesh doesn't contain position, normal or texture cord data!");
			}

			// indices
			const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
			const void* rawIndices = getIndexData(model, indexAccessor);

			// position data
			auto positionIt = getAttributeIt("POSITION", primitive.attributes);
			const tinygltf::Accessor& positionAccessor = model.accessors[positionIt->second];

			bool colorFound = (colorData);
			bool tangentFound = (tangentData);

			// calculate the tangents if theyre not found
			std::vector<dml::vec4> tangents(positionAccessor.count, dml::vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
			if (!tangentFound) {
				LOG_WARNING("Could not load tangents. Calculating tangents...");

				switch (indexAccessor.componentType) {
				case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
					calculateTangents<uint8_t>(positionData, texCoordData, tangents, rawIndices, indexAccessor.count);
					break;
				case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
					calculateTangents<uint16_t>(positionData, texCoordData, tangents, rawIndices, indexAccessor.count);
					break;
				case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
					calculateTangents<uint32_t>(positionData, texCoordData, tangents, rawIndices, indexAccessor.count);
					break;
				default:
					LOG_WARNING("Unsupported index type: " + std::to_string(indexAccessor.type));
					break;
				}

				normalizeTangents(tangents);
			}

			for (size_t i = 0; i < indexAccessor.count; i++) {
				uint32_t index;  // use the largest type to ensure no overflow

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
				tinygltf::Material& material = model.materials[primitive.material];
				Material texture;

				// base color texture
				if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
					tinygltf::TextureInfo& texInfo = material.pbrMetallicRoughness.baseColorTexture;
					tinygltf::Texture& tex = model.textures[texInfo.index];
					texture.baseColor.gltfImage = model.images[tex.source];
					texture.baseColor.path = "gltf";
					texture.baseColor.found = true;
					newObject.textureCount++;
				}

				// metallic-roughness Texture
				if (material.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
					tinygltf::TextureInfo& texInfo = material.pbrMetallicRoughness.metallicRoughnessTexture;
					tinygltf::Texture& tex = model.textures[texInfo.index];
					texture.metallicRoughness.gltfImage = model.images[tex.source];
					texture.metallicRoughness.path = "gltf";
					texture.metallicRoughness.found = true;
					newObject.textureCount++;
				}

				// normal map
				if (material.normalTexture.index >= 0) {
					tinygltf::NormalTextureInfo& texInfo = material.normalTexture;
					tinygltf::Texture& tex = model.textures[texInfo.index];
					texture.normalMap.gltfImage = model.images[tex.source];
					texture.normalMap.path = "gltf";
					texture.normalMap.found = true;
					newObject.textureCount++;
				}

				// emissive map
				if (material.emissiveTexture.index >= 0) {
					tinygltf::TextureInfo& texInfo = material.emissiveTexture;
					tinygltf::Texture& tex = model.textures[texInfo.index];
					texture.emissiveMap.gltfImage = model.images[tex.source];
					texture.emissiveMap.path = "gltf";
					texture.emissiveMap.found = true;
					newObject.textureCount++;
				}

				// occlusion map
				if (material.occlusionTexture.index >= 0) {
					tinygltf::OcclusionTextureInfo& texInfo = material.occlusionTexture;
					tinygltf::Texture& tex = model.textures[texInfo.index];
					texture.occlusionMap.gltfImage = model.images[tex.source];
					texture.occlusionMap.path = "gltf";
					texture.occlusionMap.found = true;
					newObject.textureCount++;
				}

				// ensure the model is PBR
				LOG_WARNING_IF("Model isnt PBR!!", !texture.baseColor.found && !texture.metallicRoughness.found && !texture.normalMap.found && !texture.emissiveMap.found && !texture.occlusionMap.found);
				newObject.material = texture;
			}
			LOG_WARNING_IF("Primitive " + std::to_string(primitive.material) + " doesn't have a material/texture", primitive.material < 0);
		}

		newObject.vertices = tempVertices;
		newObject.indices = tempIndices;

		size_t hash1 = std::hash<std::size_t>{}(meshInd * tempIndices.size() * tempVertices.size());
		size_t hash2 = std::hash<std::string>{}(mesh.name);

		newObject.meshHash = hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));

		newObject.name = mesh.name;

		newObject.scale = scale;
		newObject.position = pos;
		newObject.rotation = rot;

		// calculate the model matrix for the mesh
		newObject.modelMatrix = calcMeshWM(model, meshInd, parentInd, newObject);

		return newObject;
	}
};
