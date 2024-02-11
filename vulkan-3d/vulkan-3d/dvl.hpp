// Drew's Vertex Library (DVL)
// Designed to work with Vulkan and GLTF 2.0

#pragma once
#include "dml.hpp"
#include <unordered_map>
#include <unordered_set>
#include <set>
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

	struct PairHash {
		size_t operator()(const std::pair<uint32_t, uint32_t>& pair) const {
			auto hash1 = std::hash<uint32_t>{}(pair.first);
			auto hash2 = std::hash<uint32_t>{}(pair.second);
			return hash1 ^ (hash2 << 1); // combine the hashes
		}
	};

	struct Edge {
		uint32_t v1, v2; // vertex indices
		float error; // error metric for this edge

		// operator for comparing edges - directed edges
		bool operator==(const Edge& other) const {
			return v1 == other.v1 && v2 == other.v2;
		}
	};

	struct EdgeComp {
		bool operator()(const Edge& a, const Edge& b) const {
			if (a.error < b.error) return true;
			if (a.error > b.error) return false;

			if (a.v1 < b.v1) return true;
			if (a.v1 > b.v1) return false;

			return a.v2 < b.v2;
		}
	};

	struct HalfEdge {
		uint32_t vert; // vert at the end of the halfedge
		uint32_t pair; // oppositely oriented adjacent halfedge 
		uint32_t next; // next halfedge around the face
		bool boundary;
	};

	static void simplifyMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, float percent) {
		// help from: https://graphics.stanford.edu/courses/cs468-10-fall/LectureSlides/08_Simplification.pdf
		if (percent == 0 || percent > 100) {
			throw std::invalid_argument("Percent must be between 1 and 100!!!");
		}

		std::vector<HalfEdge> halfEdges = buildHalfEdges(vertices, indices);
		std::vector<dml::mat4> quadrics = calcVertQuadrics(vertices, halfEdges); // initialize all quadrics (1 per vertex)
		std::set<Edge, EdgeComp> edgeSet;

		initSet(edgeSet, vertices, halfEdges, quadrics);
		uint32_t targetHE = static_cast<uint32_t>(halfEdges.size() * (percent / 100.0f));

		uint32_t i = 0;
		while (halfEdges.size() > targetHE) {
			if (i >= 400) {
				std::cout << "Breakpoint at: " << i << std::endl;
			}
			auto start = std::chrono::high_resolution_clock::now();
			Edge bestEdge = findBestEC(edgeSet);
			collapseEdge(edgeSet, halfEdges, quadrics, bestEdge, vertices);

			std::cout << "--------------" << std::endl;
			bool right = isOrderCorrect(halfEdges);
			if (right) {
				std::cout << "Order is correct!" << std::endl;
			}
			else {
				std::cout << "Order is incorrect!" << std::endl;
			}
			std::cout << "--------------" << std::endl;


			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			std::cout << "Time taken: " << duration.count() << " microseconds. i: " << i
				<< ". Halfedge count: " << halfEdges.size() << " / " << targetHE
				<< ". set size : " << edgeSet.size() << std::endl;
			i++;
		}

		std::vector<Vertex> originalVertices = vertices;
		undoHalfEdges(halfEdges, vertices, indices, originalVertices);
	}
private:
	static void collapseEdge(std::set<Edge, EdgeComp>& edgeSet, std::vector<HalfEdge>& halfEdges,
		std::vector<dml::mat4>& quadrics, Edge& bestEdge, std::vector<Vertex>& verts) {
		uint32_t v1 = bestEdge.v1;
		uint32_t v2 = bestEdge.v2;

		// remove edges connected to v1 or v2 from edgeset
		for (auto it = edgeSet.begin(); it != edgeSet.end(); ) {
			if (it->v1 == v1 || it->v1 == v2 || it->v2 == v1 || it->v2 == v2) {
				it = edgeSet.erase(it);
			}
			else {
				++it;
			}
		}

		updateHalfedgeIndices(halfEdges, v1, v2);

		// get the new pos for the vertex
		verts[v1].pos = (verts[v1].pos + verts[v2].pos) / 2.0f;

		// update quadrics for the affected vertices
		quadrics[v1] = quadrics[v1] + quadrics[v2];

		// add newly formed edges to edgeSet
		for (uint32_t i = 0; i < halfEdges.size(); ++i) {
			const HalfEdge& h = halfEdges[i];
			if (h.vert == v1 && !h.boundary) {
				Edge newEdge;
				newEdge.v1 = v1;
				newEdge.v2 = halfEdges[h.pair].vert;
				newEdge.error = calcError(newEdge, quadrics, verts);
				edgeSet.insert(newEdge);
			}
		}
	}

	static bool correctIndex(const std::vector<HalfEdge>& halfEdges, const HalfEdge& h) {
		if (h.next >= halfEdges.size()) return false;
		HalfEdge nextHE = halfEdges[h.next];
		if (h.boundary) {
			if (h.vert == nextHE.vert) return false;
		}
		else {
			HalfEdge pairHE = halfEdges[h.pair];
			if (h.vert == nextHE.vert) return false;
			if (h.vert == pairHE.vert) return false;
			if (h.pair >= halfEdges.size()) return false;
		}
		if (h.vert == h.next || h.vert == h.pair) return false;
		return true;
	}


	static void updateHalfedgeIndices(std::vector<HalfEdge>& halfEdges, uint32_t v1, uint32_t v2) {
		std::vector<uint32_t> indexmap(halfEdges.size(), UINT32_MAX);
		uint32_t newIndex = 0;

		// reindex valid halfedges and find ones to remove
		for (uint32_t i = 0; i < halfEdges.size(); ++i) {
			HalfEdge& h = halfEdges[i];

			// dont process this halfedge bc it will be removed
			if (h.vert == v2 || h.next == v2 || h.pair == v2) {
				continue;
			}
			if (!correctIndex(halfEdges, h)) {
				continue;
			}
			indexmap[i] = newIndex++; // store the new index for valid halfedges
		}

		// compact the halfEdges vector and update indicies
		uint32_t compactIndex = 0; // next valid position in halfedges
		for (uint32_t i = 0; i < halfEdges.size(); ++i) {
			if (indexmap[i] != UINT32_MAX) { // if not marked for removal
				HalfEdge& h = halfEdges[i];
				// if marked for removal, update the halfedge indices
				h.next = indexmap[h.next] != UINT32_MAX ? indexmap[h.next] : h.next;
				h.pair = indexmap[h.pair] != UINT32_MAX ? indexmap[h.pair] : h.pair;

				// move to compact position if i isnt the compact index
				if (i != compactIndex) {
					halfEdges[compactIndex] = h;
				}
				compactIndex++;
			}
		}
		// resize to remove the unused tail elements
		// this is pretty fast
		halfEdges.resize(compactIndex);
	}

	// find the best edge to collapse and remove invalid edges from the set
	static Edge findBestEC(std::set<Edge, EdgeComp>& edgeSet) {
		if (edgeSet.empty()) {
			throw std::runtime_error("No valid edges left in the set!");
		}
		Edge bestEdge = *edgeSet.begin();
		std::cout << bestEdge.error << std::endl;
		edgeSet.erase(edgeSet.begin());
		return bestEdge;
	}

	// check if the halfedges are in the correct order
	static bool isOrderCorrect(const std::vector<HalfEdge>& halfEdges) {
		if (halfEdges.empty()) return true;

		// check if halfedges exceed bounds
		for (size_t i = 0; i < halfEdges.size(); ++i) {
			const HalfEdge& he = halfEdges[i];
			if (!he.boundary) {
				if (he.next >= halfEdges.size() || he.pair >= halfEdges.size()) {
					std::cerr << "Halfedge index out of bounds!" << std::endl;
					return false;
				}
			}
			else {
				if (he.next >= halfEdges.size()) {
					std::cerr << "Boundary halfedge index out of bounds!" << std::endl;
					return false;
				}
			}
		}

		// check if halfedges form a loop
		for (const auto& he : halfEdges) {
			if (!he.boundary) {
				const HalfEdge& nextHE = halfEdges[he.next];
				const HalfEdge& pairHE = halfEdges[he.pair];

				if (he.vert == nextHE.vert || he.vert == pairHE.vert) {
					std::cerr << "Halfedges form a loop!" << std::endl;
					return false;
				}
			}
			else {
				// only check the next halfedge for boundary edges
				const HalfEdge& nextHE = halfEdges[he.next];

				if (he.vert == nextHE.vert) {
					std::cerr << "Boundary halfedge form a loop!" << std::endl;
					return false;
				}
			}
		}
		return true;
	}


	// convert the array of vertices and indicies into a half edge data structure
	static std::vector<HalfEdge> buildHalfEdges(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
		std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, PairHash> edgeMap;

		std::vector<HalfEdge> halfEdges(indices.size());
		for (uint32_t i = 0; i < indices.size(); i += 3) {
			for (uint32_t j = 0; j < 3; ++j) {
				uint32_t currentIndex = i + j;
				uint32_t nextIndex = i + ((j + 1) % 3);

				uint32_t v1 = indices[currentIndex];
				uint32_t v2 = indices[nextIndex];

				halfEdges[currentIndex].vert = v2;
				halfEdges[currentIndex].next = (currentIndex % 3 == 2) ? (currentIndex - 2) : (currentIndex + 1);
				halfEdges[currentIndex].boundary = true;

				auto edgePair = std::make_pair(v1, v2);
				edgeMap[edgePair] = currentIndex;

				auto it = edgeMap.find(std::make_pair(v2, v1));
				if (it != edgeMap.end()) {
					uint32_t twinIndex = it->second;
					halfEdges[currentIndex].pair = twinIndex;
					halfEdges[twinIndex].pair = currentIndex;
					halfEdges[currentIndex].boundary = false;
					halfEdges[twinIndex].boundary = false;
				}
			}
		}
		uint32_t b = 0;
		for (const auto& h : halfEdges) {
			if (h.boundary) b++;
		}
		std::cout << "Boundary count: " << b << std::endl;
		std::cout << "Percent: " << (b / static_cast<float>(halfEdges.size())) * 100.0f << "%" << std::endl;


		return halfEdges;
	}

	// function to convert the halfedges back into vertices and indices
	static void undoHalfEdges(const std::vector<HalfEdge>& halfEdges, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
		const std::vector<Vertex>& ov) {
		vertices.clear();
		indices.clear();

		std::unordered_map<uint32_t, uint32_t> vertexMap;
		std::unordered_set<uint32_t> processed;

		for (uint32_t i = 0; i < halfEdges.size(); ++i) {
			if (processed.find(i) != processed.end()) {
				continue;
			}

			uint32_t start = i;
			uint32_t current = start;
			do {
				if (current >= halfEdges.size() || halfEdges[current].vert >= ov.size()) {
					std::cerr << "halfEdges out of range!!!" << std::endl;
					break;
				}

				uint32_t vertIndex = halfEdges[current].vert;
				if (vertexMap.find(vertIndex) == vertexMap.end()) {
					vertexMap[vertIndex] = static_cast<uint32_t>(vertices.size()); // add vertex if it hasnt been added before
					vertices.push_back(ov[vertIndex]);
				}
				indices.push_back(vertexMap[vertIndex]);
				processed.insert(current);
				current = halfEdges[current].next;

			} while (current != start && processed.find(current) == processed.end());
		}
	}


	static dml::mat4 calcFaceQuadric(const HalfEdge& h1, const HalfEdge& h2, const HalfEdge& h3, const std::vector<Vertex>& vertices) {
		const Vertex& v1 = vertices[h1.vert];
		const Vertex& v2 = vertices[h2.vert];
		const Vertex& v3 = vertices[h3.vert];

		dml::vec3 normal = dml::cross(v2.pos - v1.pos, v3.pos - v1.pos);
		normal = dml::normalize(normal);
		float d = dml::dot(normal, v1.pos);

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

	static float calcError(const Edge& edge, const std::vector<dml::mat4>& quadrics, const std::vector<Vertex>& verts) {
		const dml::mat4& q1 = quadrics[edge.v1];
		const dml::mat4& q2 = quadrics[edge.v2];

		// homogeneous representation of the edges vertices
		dml::vec4 v1(verts[edge.v1].pos, 1.0f);
		dml::vec4 v2(verts[edge.v2].pos, 1.0f);

		// calc error for each vertex
		float e1 = dml::dot(v1, q1 * v1);
		float e2 = dml::dot(v2, q2 * v2);

		float total = e1 + e2;
		return total;
	}


	static void initSet(std::set<Edge, EdgeComp>& s, const std::vector<Vertex>& vertices, const std::vector<HalfEdge>& halfEdges, const std::vector<dml::mat4>& quadrics) {

		// temporary set to ensure each edge is only processed once
		std::unordered_set<std::pair<uint32_t, uint32_t>, PairHash> c;

		for (const HalfEdge& h : halfEdges) {
			uint32_t v1 = h.vert;
			uint32_t v2 = halfEdges[h.pair].vert;

			// ensure each edge is only processed once
			if (v1 > v2) std::swap(v1, v2);
			if (c.find(std::make_pair(v1, v2)) != c.end()) continue;
			c.insert(std::make_pair(v1, v2));

			Edge e;
			e.v1 = v1;
			e.v2 = v2;

			// compute the edge collapse error
			if (h.boundary) {
				e.error = std::numeric_limits<float>::max();
			}
			else {
				e.error = calcError(e, quadrics, vertices);
			}

			s.insert(e);
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