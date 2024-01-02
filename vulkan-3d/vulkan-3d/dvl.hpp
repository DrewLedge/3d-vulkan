// Drew's Vertex Library (DVL)
// Designed to work with Vulkan and GLTF 2.0

#pragma once
#include "dml.hpp"
#include <queue>
#ifndef DVL_H
#define DVL_H

class dvl {
public:
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

	static void calculateTangents(const float* positionData, const float* texCoordData, std::vector<dml::vec4>& tangents, const void* rawIndices, size_t size) {
		for (size_t i = 0; i < size; i += 3) {
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
	}

	static void normalizeTangents(std::vector<dml::vec4>& tangents) {
		for (dml::vec4& tangent : tangents) {
			dml::vec3 normalizedTangent = dml::normalize(tangent.xyz());
			tangent.x = normalizedTangent.x;
			tangent.y = normalizedTangent.y;
			tangent.z = normalizedTangent.z;
		}
	}

	struct Edge {
		uint32_t v1, v2; // vertex indices
		float error; // error metric for this edge

		bool operator>(const Edge& other) const {
			return error > other.error;
		}
	};

	static void simplifyMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, float percent) {
		if (percent == 0 || percent > 100) {
			throw std::invalid_argument("Percent must be between 1 and 100!!!");
		}
	}
private:
	static uint64_t encodeEdgeKey(uint32_t v1, uint32_t v2) { // encodes two vertex indices into a single 64-bit integer
		return static_cast<uint64_t>(v1) << 32 | v2;
	}

	static std::pair<uint32_t, uint32_t> decodeEdgeKey(uint64_t key) { // decodes a 64-bit integer into two vertex indices
		return { static_cast<uint32_t>(key >> 32), static_cast<uint32_t>(key & 0xFFFFFFFF) };
	}

	dml::mat4 calcFaceQuadric(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
		dml::vec3 normal = dml::cross(v2.pos - v1.pos, v3.pos - v1.pos);
		normal = dml::normalize(normal);
		float d = -dml::dot(normal, v1.pos);

		dml::vec4 plane(normal, d);
		dml::mat4 quadric = dml::outerProduct(plane, plane);

		return quadric;
	}

	std::vector<dml::mat4> calcVertQuadrics(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
		std::vector<dml::mat4> quadrics(vertices.size(), dml::mat4(0.0f));

		for (size_t i = 0; i < indices.size(); i += 3) {
			dml::mat4 faceQuadric = calcFaceQuadric(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]);
			quadrics[indices[i]] += faceQuadric;
			quadrics[indices[i + 1]] += faceQuadric;
			quadrics[indices[i + 2]] += faceQuadric;
		}

		return quadrics;
	}

	float calcVertError(const dml::mat4& quadric, const dml::vec3& pos) {
		dml::vec4 pos4(pos, 1.0f);
		return dml::dot(quadric * pos4, pos4);
	}

};

#endif
;