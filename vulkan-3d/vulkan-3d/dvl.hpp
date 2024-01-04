// Drew's Vertex Library (DVL)
// Designed to work with Vulkan and GLTF 2.0

#pragma once
#include "dml.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>
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

	struct PairHash {
		size_t operator()(const std::pair<uint32_t, uint32_t>& pair) const {
			auto hash1 = std::hash<uint32_t>{}(pair.first);
			auto hash2 = std::hash<uint32_t>{}(pair.second);
			return hash1 ^ (hash2 << 1); // combine the hashes
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

	struct HalfEdge {
		uint32_t vert; // vert at the end of the halfedge
		uint32_t pair; // oppositely oriented adjacent halfedge 
		uint32_t face; // face the halfedge borders
		uint32_t next; // next halfedge around the face
	};

	static void simplifyMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, float percent) {
		// help from: https://graphics.stanford.edu/courses/cs468-10-fall/LectureSlides/08_Simplification.pdf
		if (percent == 0 || percent > 100) {
			throw std::invalid_argument("Percent must be between 1 and 100!!!");
		}
		std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> queue;

		std::vector<HalfEdge> halfEdges = buildHalfEdges(vertices, indices);
		std::vector<dml::mat4> quadrics = calcVertQuadrics(vertices, halfEdges); // initialize all quadrics (1 per vertex)
		initQueue(queue, vertices, halfEdges, quadrics);

		//size_t targetVertices = static_cast<size_t>(vertices.size() * (percent / 100));
		//while (halfEdges.size() / 2 > targetVertices) {
		//	Edge bestEdge = findBestEC(queue);
		//	//collapseEdge(vertices, indices, bestEdge, queue, quadrics);
		//	//std::cout << vertices.size() << " / " << targetVertices << std::endl;
		//}
		std::vector<Vertex> originalVertices = vertices;
		undoHalfEdges(halfEdges, vertices, indices, originalVertices);
	}
private:
	// convert the array of vertices and indicies into a half edge data structure
	static std::vector<HalfEdge> buildHalfEdges(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
		std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, PairHash> edgeMap;
		std::vector<HalfEdge> halfEdges(indices.size());

		for (uint32_t i = 0; i < indices.size(); i += 3) {
			for (uint32_t j = 0; j < 3; ++j) {
				uint32_t currentIndex = i + j;
				uint32_t nextIndex = i + (j + 1) % 3;

				uint32_t vert1 = indices[currentIndex];
				uint32_t vert2 = indices[nextIndex];

				halfEdges[currentIndex].vert = vert2;
				halfEdges[currentIndex].face = currentIndex / 3;
				halfEdges[currentIndex].next = (currentIndex % 3 == 2) ? (currentIndex - 2) : (currentIndex + 1);

				auto edgePair = std::make_pair(vert1, vert2);
				edgeMap[edgePair] = currentIndex;

				auto it = edgeMap.find(std::make_pair(vert2, vert1));
				if (it != edgeMap.end()) {
					uint32_t twinIndex = it->second;
					halfEdges[currentIndex].pair = twinIndex;
					halfEdges[twinIndex].pair = currentIndex;
				}
			}
		}

		return halfEdges;
	}

	// function to convert a halfedge back into vertices and indices
	static void undoHalfEdges(const std::vector<HalfEdge>& halfEdges, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
		const std::vector<Vertex>& ov) {
		vertices.clear();
		indices.clear();

		std::unordered_set<uint32_t> processedFaces;
		std::unordered_map<uint32_t, uint32_t> vertexMap;

		for (const HalfEdge& h : halfEdges) {
			if (processedFaces.find(h.face) != processedFaces.end()) continue;
			processedFaces.insert(h.face);

			uint32_t ind = &h - &halfEdges[0]; // current index of the halfedge
			for (int i = 0; i < 3; ++i) {
				uint32_t vertIndex = halfEdges[ind].vert;
				if (vertexMap.find(vertIndex) == vertexMap.end()) {
					// add vertex if it hasnt been added before
					vertexMap[vertIndex] = vertices.size();
					vertices.push_back(ov[vertIndex]);
				}
				indices.push_back(vertexMap[vertIndex]);
				ind = halfEdges[ind].next;
			}
		}
	}


	static dml::mat4 calcFaceQuadric(const HalfEdge& h1, const HalfEdge& h2, const HalfEdge& h3, const std::vector<Vertex>& vertices) {
		const Vertex& v1 = vertices[h1.vert];
		const Vertex& v2 = vertices[h2.vert];
		const Vertex& v3 = vertices[h3.vert];

		dml::vec3 normal = dml::cross(v2.pos - v1.pos, v3.pos - v1.pos);
		normal = dml::normalize(normal);
		float d = -dml::dot(normal, v1.pos);

		dml::vec4 plane(normal, d);
		dml::mat4 quadric = dml::outerProduct(plane, plane);

		return quadric;
	}

	static std::vector<dml::mat4> calcVertQuadrics(const std::vector<Vertex>& vertices, const std::vector<HalfEdge>& halfEdges) {
		std::vector<dml::mat4> quadrics(vertices.size(), dml::mat4(0.0f));

		for (size_t i = 0; i < halfEdges.size(); i += 3) {
			dml::mat4 faceQuadric = calcFaceQuadric(halfEdges[i], halfEdges[i + 1], halfEdges[i + 2], vertices);
			quadrics[halfEdges[i].vert] += faceQuadric;
			quadrics[halfEdges[i + 1].vert] += faceQuadric;
			quadrics[halfEdges[i + 2].vert] += faceQuadric;
		}

		return quadrics;
	}

	static float calcVertError(const dml::mat4& quadric, const dml::vec3& pos) {
		dml::vec4 pos4(pos, 1.0f);
		return dml::dot(quadric * pos4, pos4);
	}

	static Edge findBestEC(std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>>& q) { // find the best edge to collapse
		Edge bestEdge = q.top();
		q.pop();
		return bestEdge;
	}

	static void initQueue(std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>>& q, const std::vector<Vertex>& vertices, const std::vector<HalfEdge>& halfEdges, const std::vector<dml::mat4>& quadrics) {
		std::unordered_set<std::pair<uint32_t, uint32_t>, PairHash> c;

		for (const auto& halfEdge : halfEdges) {
			uint32_t v1 = halfEdge.vert;
			uint32_t v2 = halfEdges[halfEdge.pair].vert;

			// ensure each edge is only processed once
			if (v1 > v2) std::swap(v1, v2);
			if (c.find(std::make_pair(v1, v2)) != c.end()) continue;
			c.insert(std::make_pair(v1, v2));

			Edge e;
			e.v1 = v1;
			e.v2 = v2;

			// compute the edge collapse error
			dml::mat4 combinedQuadric = quadrics[v1] + quadrics[v2];
			dml::vec3 midPoint = (vertices[v1].pos + vertices[v2].pos) / 2.0f;
			e.error = calcVertError(combinedQuadric, midPoint);

			q.push(e);
		}
	}

	static void updateVertAttributes(Vertex& vertex, const Vertex& other) {
		vertex.tex = (vertex.tex + other.tex) / 2.0f;
		vertex.col = (vertex.col + other.col) / 2.0f;
		vertex.normal = (vertex.normal + other.normal) / 2.0f;
		vertex.alpha = (vertex.alpha + other.alpha) / 2.0f;
		vertex.tangent = (vertex.tangent + other.tangent) / 2.0f;

	}

};

#endif
;