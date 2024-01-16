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

		bool operator>(const Edge& other) const {
			return error > other.error;
		}

		bool operator==(const Edge& other) const {
			return (v1 == other.v1 && v2 == other.v2) || (v1 == other.v2 && v2 == other.v1);
		}
	};

	struct EdgeComp { //comparator for the set of edges
		static constexpr double EPSILON = 1e-6;

		bool operator()(const Edge& a, const Edge& b) const {
			if (a.v1 != b.v1) return a.v1 < b.v1;
			if (a.v2 != b.v2) return a.v2 < b.v2;

			return (a.error < b.error - EPSILON);
		}
	};

	struct HalfEdge {
		uint32_t vert; // vert at the end of the halfedge
		uint32_t pair; // oppositely oriented adjacent halfedge 
		uint32_t next; // next halfedge around the face
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
		uint32_t targetVerts = static_cast<uint32_t>(vertices.size() * (percent / 100.0f));

		int i = 0;
		while (vertices.size() > targetVerts) {
			if (i >= 41) {
				std::cout << "Breakpoint at: " << i << std::endl;
			}
			auto start = std::chrono::high_resolution_clock::now();
			Edge bestEdge = findBestEC(edgeSet);
			collapseEdge(edgeSet, halfEdges, quadrics, bestEdge, vertices);
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

			std::cout << "Time taken: " << duration.count() << " microseconds. i: " << i
				<< ". Halfedge count: " << halfEdges.size() << ". Vertex count: " << vertices.size()
				<< "/" << targetVerts << ". set size: " << edgeSet.size() << std::endl;
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

		auto deleteSet = [&](uint32_t vertex) {
			std::vector<uint32_t> indexes = getConnectedHalfEdgeIndices(vertex, halfEdges);
			for (uint32_t index : indexes) {
				HalfEdge& h = halfEdges[index];

				Edge e;
				e.v1 = h.vert;
				e.v2 = halfEdges[h.pair].vert;

				// skip the edge that is being collapsed
				// this is because it has already been removed from the set
				if (e == bestEdge) continue;

				dml::mat4 combinedQ = quadrics[e.v1] + quadrics[e.v2];
				dml::vec3 mp = (verts[e.v1].pos + verts[e.v2].pos) / 2.0f;
				e.error = calcVertError(combinedQ, mp);


				auto it = edgeSet.find(e);
				if (it != edgeSet.end()) {
					std::cout << "Found edge in set" << std::endl;
				}
				else {
					std::cout << "Did not find edge in set" << std::endl;
				}
				edgeSet.erase(e);
			}
			};

		// delete all the edges from the set that will be affected by the collapse
		// this is so the edges can be updated and reinserted later
		deleteSet(v1);
		deleteSet(v2);

		updateVertAttributes(verts[v1], verts[v2]);

		// get the new pos for the vertex
		verts[v1].pos = (verts[v1].pos + verts[v2].pos) / 2.0f;

		// update quadrics for the affected vertices
		quadrics[v1] = quadrics[v1] + quadrics[v2];

		// for each halfedge around v2, update the vertex to v1
		std::vector<uint32_t> v2ind = getConnectedHalfEdgeIndices(v2, halfEdges);
		for (uint32_t index : v2ind) {
			HalfEdge& h = halfEdges[index];
			if (halfEdges[h.pair].vert == v1) {
				halfEdges[h.pair].pair = halfEdges[h.next].pair;
				halfEdges[halfEdges[h.next].pair].pair = h.pair;
			}
			else {
				h.vert = v1;
			}
		}

		// remove redundant halfedges
		halfEdges.erase(
			std::remove_if(halfEdges.begin(), halfEdges.end(),
				[&](const HalfEdge& h) { return h.vert == h.next || h.vert == v2; }),
			halfEdges.end());

		// update remaining half-edge indices
		for (auto& h : halfEdges) {
			if (h.vert > v2) --h.vert;
			if (h.pair > v2) --h.pair;
			if (h.next > v2) --h.next;
		}

		uint32_t swappedVertInd = static_cast<uint32_t>(verts.size() - 1);
		if (v2 != swappedVertInd) {
			std::swap(verts[v2], verts[swappedVertInd]);
			std::swap(quadrics[v2], quadrics[swappedVertInd]);

			// update half edges that referenced the swapped vertex
			for (auto& h : halfEdges) {
				if (h.vert == swappedVertInd) h.vert = v2;
				if (h.pair == swappedVertInd) h.pair = v2;
				if (h.next == swappedVertInd) h.next = v2;
			}
		}
		verts.pop_back();
		quadrics.pop_back();

		bool right = isOrderCorrect(halfEdges);
		if (right) std::cout << "Order is correct!" << std::endl;
		else std::cout << "Order is incorrect!" << std::endl;

		// update all edges connected to v1 and add them to the set
		std::vector<uint32_t> v1ind = getConnectedHalfEdgeIndices(v1, halfEdges);
		for (uint32_t index : v1ind) {
			HalfEdge& h = halfEdges[index];

			Edge e;
			e.v1 = h.vert;
			e.v2 = halfEdges[h.pair].vert;

			// dont add the edge that was just collapsed
			if (e == bestEdge) continue;

			dml::mat4 combinedQ = quadrics[e.v1] + quadrics[e.v2];
			dml::vec3 mp = (verts[e.v1].pos + verts[e.v2].pos) / 2.0f;
			e.error = calcVertError(combinedQ, mp);

			edgeSet.insert(e);
		}
	}

	static bool isOrderCorrect(const std::vector<HalfEdge>& halfEdges) {
		if (halfEdges.empty()) return true;

		// check if halfedges exceed bounds
		for (size_t i = 0; i < halfEdges.size(); ++i) {
			const HalfEdge& he = halfEdges[i];
			if (he.next >= halfEdges.size() || he.pair >= halfEdges.size()) {
				return false;
			}
		}

		// check if halfedges form a loop
		for (const auto& he : halfEdges) {
			const HalfEdge& nextHE = halfEdges[he.next];
			const HalfEdge& pairHE = halfEdges[he.pair];

			if (he.vert == nextHE.vert || he.vert == pairHE.vert) {
				return false;
			}
		}
		return true;
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

	// get the indices of halfedges connected to a vertex
	static std::vector<uint32_t> getConnectedHalfEdgeIndices(uint32_t vertex, const std::vector<HalfEdge>& halfEdges) {
		std::vector<uint32_t> connectedHalfEdgeInd;
		std::unordered_set<uint32_t> visited;

		// find a starting halfedge index connected to the vertex
		uint32_t start = std::numeric_limits<uint32_t>::max();
		for (uint32_t i = 0; i < halfEdges.size(); ++i) {
			if (halfEdges[i].vert == vertex) {
				start = i;
				break;
			}
		}

		// if no starting halfedge index was found, return an empty vector
		if (start == std::numeric_limits<uint32_t>::max()) {
			std::cerr << "No starting halfedge found for vertex " << vertex << std::endl;
			return connectedHalfEdgeInd;
		}

		uint32_t current = start;
		do {
			// if this halfedge was already visited, break out of the loop
			if (visited.find(current) != visited.end()) break;

			// add the current halfedge to the list of connected halfedges
			connectedHalfEdgeInd.push_back(current);

			// mark the current halfedge as visited
			visited.insert(current);

			// move to the pair of the next halfedge
			uint32_t nextIndex = halfEdges[current].next;

			uint32_t pairIndex = halfEdges[nextIndex].pair;
			//std::cout << "next: " << nextIndex << " pair: " << pairIndex << " size: " << halfEdges.size() << std::endl;
			current = pairIndex;

		} while (current != start);

		return connectedHalfEdgeInd;
	}

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

	// function to convert the halfedges back into vertices and indices
	static void undoHalfEdges(const std::vector<HalfEdge>& halfEdges, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
		const std::vector<Vertex>& ov) {
		vertices.clear();
		indices.clear();

		std::unordered_map<uint32_t, uint32_t> vertexMap;

		for (const HalfEdge& h : halfEdges) {
			uint32_t ind = &h - &halfEdges[0]; // current index of the halfedge
			for (int i = 0; i < 3; ++i) {
				uint32_t vertIndex = halfEdges[ind].vert;
				if (vertexMap.find(vertIndex) == vertexMap.end()) {
					// add vertex if it hasnt been added before
					vertexMap[vertIndex] = static_cast<uint32_t>(vertices.size());
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

	static void initSet(std::set<Edge, EdgeComp>& s, const std::vector<Vertex>& vertices, const std::vector<HalfEdge>& halfEdges, const std::vector<dml::mat4>& quadrics) {

		// temporary set to ensure each edge is only processed once
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