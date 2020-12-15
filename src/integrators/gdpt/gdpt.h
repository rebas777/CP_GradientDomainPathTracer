#include <mitsuba/mitsuba.h>
#include <mitsuba/render/scene.h>
#include "gdpt_wr.h"

MTS_NAMESPACE_BEGIN

/// Classification of vertices into diffuse and glossy.
enum VertexType {
	VERTEX_TYPE_GLOSSY,     ///< "Specular" vertex that requires the half-vector duplication shift.
	VERTEX_TYPE_DIFFUSE     ///< "Non-specular" vertex that is rough enough for the reconnection shift.
};

enum RayConnection {
	RAY_NOT_CONNECTED,      ///< Not yet connected - shifting in progress.
	RAY_RECENTLY_CONNECTED, ///< Connected, but different incoming direction so needs a BSDF evaluation.
	RAY_CONNECTED           ///< Connected, allows using BSDF values from the base path.
};

struct GDPTConfiguration {
	int m_maxDepth;
	double m_shiftThreshold;

	inline GDPTConfiguration() { }

	inline GDPTConfiguration(Stream* stream) {
		m_maxDepth = stream->readInt();
		m_shiftThreshold = (double)stream->readFloat();
	}

	inline void serialize(Stream* stream) const {
		stream->writeInt(m_maxDepth);
		stream->writeFloat(m_shiftThreshold);
	}
};

struct RayState {
	RayState() :
		radiance(0.0f),
		gradient(0.0f),
		eta(1.0f),
		pdf(1.0f),
		throughput(Spectrum(0.0f)),
		alive(true),
		connection_status(RAY_NOT_CONNECTED)
	{}

	/// Adds radiance to the ray.
	inline void addRadiance(const Spectrum& contribution, double weight) {
		Spectrum color = contribution * weight;
		radiance += color;
	}

	/// Adds gradient to the ray.
	inline void addGradient(const Spectrum& contribution, double weight) {
		Spectrum color = contribution * weight;
		gradient += color;
	}

	RayDifferential ray;             ///< Current ray.

	Spectrum throughput;             ///< Current throughput of the path.
	double pdf;                       ///< Current PDF of the path.

	// Note: Instead of storing throughput and pdf, it is possible to store Veach-style weight (throughput divided by pdf), if relative PDF (offset_pdf divided by base_pdf) is also stored. This might be more stable numerically.

	Spectrum radiance;               ///< Radiance accumulated so far.
	Spectrum gradient;               ///< Gradient accumulated so far.

	RadianceQueryRecord rRec;        ///< The radiance query record for this ray.
	double eta;                       ///< Current refractive index of the ray.
	bool alive;                      ///< Whether the path matching to the ray is still good. Otherwise it's an invalid offset path with zero PDF and throughput.

	RayConnection connection_status; ///< Whether the ray has been connected to the base path, or is in progress.
};

/// Stores the results of a BSDF sample.
/// Do not confuse with Mitsuba's BSDFSamplingRecord.
struct BSDFSampleResult {
	BSDFSamplingRecord bRec;  ///< The corresponding BSDF sampling record.
	Spectrum weight;          ///< BSDF weight of the sampled direction.
	double pdf;                ///< PDF of the BSDF sample.
};

/// Result of a reconnection shift.
struct ReconnectionShiftResult {
	bool success;   ///< Whether the shift succeeded.
	double jacobian; ///< Local Jacobian determinant of the shift.
	Vector3 wo;     ///< World space outgoing vector for the shift.
};

struct HalfVectorShiftResult {
	bool success;   ///< Whether the shift succeeded.
	double jacobian; ///< Local Jacobian determinant of the shift.
	Vector3 wo;     ///< Tangent space outgoing vector for the shift.
};

class GDPTIntegrator : public MonteCarloIntegrator {

public:

	// Initialize the integrator with the specified properties
	GDPTIntegrator(const Properties& props);

	// Unserialize from a binary data stream
	GDPTIntegrator(Stream* stream, InstanceManager* manager);

	// Serialize to a binary data stream
	void serialize(Stream* stream, InstanceManager* manager) const;

	// Query for an unbiased estimate of the radiance along <tt>r</tt>
	Spectrum Li(const RayDifferential& r, RadianceQueryRecord& rRec) const;

	/// Starts the rendering process.
	bool render(Scene* scene,
		RenderQueue* queue, const RenderJob* job,
		int sceneResID, int sensorResID, int samplerResID);


	/// Renders a block in the image.
	void renderBlock(const Scene* scene, const Sensor* sensor, Sampler* sampler, GDPTWorkResult* block,
		const bool& stop, const std::vector< TPoint2<uint8_t> >& points) const;

	void serialize(Stream* stream, InstanceManager* manager) const;
	std::string toString() const;

	MTS_DECLARE_CLASS()

private:

	Spectrum m_color = Spectrum(1);

	GDPTConfiguration m_config;

};

class GradientPathTracer {
public:

	GradientPathTracer(const Scene* scene, const Sensor* sensor, Sampler* sampler, GDPTWorkResult* gdptWorkResult, GDPTConfiguration* config)
		: m_scene(scene), m_sensor(sensor), m_sampler(sampler), m_block(gdptWorkResult), m_config(config) {}

	inline std::pair<double, double> Weighting(const double rayPdf, const double lightSamplePdf, const double bsdfSamplePdf);

	inline BSDFSampleResult sampleBSDF(RayState& rayState);

	void evaluatePoint(RadianceQueryRecord& rRec, const Point2& samplePosition, const Point2& apertureSample, double timeSample, double differentialScaleFactor,
		Spectrum& out_very_direct, Spectrum& out_throughput, Spectrum* out_gradients, Spectrum* out_neighborThroughputs);

	void evaluate(RayState& main, RayState* shiftedRays, int secondaryCount, Spectrum& out_veryDirect);


private:
	const Scene* m_scene;
	const Sensor* m_sensor;
	Sampler* m_sampler;
	GDPTWorkResult* m_block;
	const GDPTConfiguration* m_config;
};


/// Returns the vertex type of a vertex by its roughness value.
VertexType getVertexTypeByRoughness(double roughness, const GDPTConfiguration& config) {
	if (roughness <= config.m_shiftThreshold) {
		return VERTEX_TYPE_GLOSSY;
	}
	else {
		return VERTEX_TYPE_DIFFUSE;
	}
};

/// Returns the vertex type (diffuse / glossy) of a vertex, for the purposes of determining
/// the shifting strategy.
///
/// A bare classification by roughness alone is not good for multi-component BSDFs since they
/// may contain a diffuse component and a perfect specular component. If the base path
/// is currently working with a sample from a BSDF's smooth component, we don't want to care
/// about the specular component of the BSDF right now - we want to deal with the smooth component.
///
/// For this reason, we vary the classification a little bit based on the situation.
/// This is perfectly valid, and should be done.
VertexType getVertexType(const BSDF* bsdf, Intersection& its, const GDPTConfiguration& config, unsigned int bsdfType) {
	// Return the lowest roughness value of the components of the vertex's BSDF.
	// If 'bsdfType' does not have a delta component, do not take perfect speculars (zero roughness) into account in this.

	double lowest_roughness = std::numeric_limits<double>::infinity();

	bool found_smooth = false;
	bool found_dirac = false;
	for (int i = 0, component_count = bsdf->getComponentCount(); i < component_count; ++i) {
		double component_roughness = bsdf->getRoughness(its, i);

		if (component_roughness == double(0)) {
			found_dirac = true;
			if (!(bsdfType & BSDF::EDelta)) {
				// Skip Dirac components if a smooth component is requested.
				continue;
			}
		}
		else {
			found_smooth = true;
		}

		if (component_roughness < lowest_roughness) {
			lowest_roughness = component_roughness;
		}
	}

	// Roughness has to be zero also if there is a delta component but no smooth components.
	if (!found_smooth && found_dirac && !(bsdfType & BSDF::EDelta)) {
		lowest_roughness = double(0);
	}

	return getVertexTypeByRoughness(lowest_roughness, config);
};
//
//VertexType getVertexType(RayState& ray, const GDPTConfiguration& config, unsigned int bsdfType) {
//	const BSDF* bsdf = ray.rRec.its.getBSDF(ray.ray);
//	return getVertexType(bsdf, ray.rRec.its, config, bsdfType);
//};

/// Returns whether point1 sees point2.
bool testVisibility(const Scene* scene, const Point3& point1, const Point3& point2, double time) {
	Ray shadowRay;
	shadowRay.setTime(time);
	shadowRay.setOrigin(point1);
	shadowRay.setDirection(point2 - point1);
	shadowRay.mint = Epsilon;
	shadowRay.maxt = (double)1.0 - ShadowEpsilon;

	return !scene->rayIntersect(shadowRay);
}

/// Returns whether the given ray sees the environment.
bool testEnvironmentVisibility(const Scene* scene, const Ray& ray) {
	const Emitter* env = scene->getEnvironmentEmitter();
	if (!env) {
		return false;
	}

	Ray shadowRay(ray);
	shadowRay.setTime(ray.time);
	shadowRay.setOrigin(ray.o);
	shadowRay.setDirection(ray.d);

	DirectSamplingRecord directSamplingRecord;
	env->fillDirectSamplingRecord(directSamplingRecord, shadowRay);

	shadowRay.mint = Epsilon;
	shadowRay.maxt = ((double)1.0 - ShadowEpsilon) * directSamplingRecord.dist;

	return !scene->rayIntersect(shadowRay);
}

/// Tries to connect the offset path to a specific vertex of the main path.
ReconnectionShiftResult reconnectShift(const Scene* scene, Point3 mainSourceVertex, Point3 targetVertex, Point3 shiftSourceVertex, Vector3 targetNormal, Float time) {
	ReconnectionShiftResult result;

	// Check visibility of the connection.
	if (!testVisibility(scene, shiftSourceVertex, targetVertex, time)) {
		// Since this is not a light sample, we cannot allow shifts through occlusion.
		result.success = false;
		return result;
	}

	// Calculate the Jacobian.
	Vector3 mainEdge = mainSourceVertex - targetVertex;
	Vector3 shiftedEdge = shiftSourceVertex - targetVertex;

	double mainEdgeLengthSquared = mainEdge.lengthSquared();
	double shiftedEdgeLengthSquared = shiftedEdge.lengthSquared();

	Vector3 shiftedWo = -shiftedEdge / sqrt(shiftedEdgeLengthSquared);

	double mainOpposingCosine = dot(mainEdge, targetNormal) / sqrt(mainEdgeLengthSquared);
	double shiftedOpposingCosine = dot(shiftedWo, targetNormal);

	double jacobian = std::abs(shiftedOpposingCosine * mainEdgeLengthSquared) / (D_EPSILON + std::abs(mainOpposingCosine * shiftedEdgeLengthSquared));

	// Return the results.
	result.success = true;
	result.jacobian = jacobian;
	result.wo = shiftedWo;
	return result;
}

/// Tries to connect the offset path to a the environment emitter.
ReconnectionShiftResult environmentShift(const Scene* scene, const Ray& mainRay, Point3 shiftSourceVertex) {
	const Emitter* env = scene->getEnvironmentEmitter();

	ReconnectionShiftResult result;

	// Check visibility of the environment.
	Ray offsetRay = mainRay;
	offsetRay.setOrigin(shiftSourceVertex);

	if (!testEnvironmentVisibility(scene, offsetRay)) {
		// Environment was occluded.
		result.success = false;
		return result;
	}

	// Return the results.
	result.success = true;
	result.jacobian = double(1);
	result.wo = mainRay.d;

	return result;
}

/// Calculates the outgoing direction of a shift by duplicating the local half-vector.
HalfVectorShiftResult halfVectorShift(Vector3 tangentSpaceMainWi, Vector3 tangentSpaceMainWo, Vector3 tangentSpaceShiftedWi, double mainEta, double shiftedEta) {
	HalfVectorShiftResult result;

	if (Frame::cosTheta(tangentSpaceMainWi) * Frame::cosTheta(tangentSpaceMainWo) < (double)0) {
		// Refraction.

		// Refuse to shift if one of the Etas is exactly 1. This causes degenerate half-vectors.
		if (mainEta == (double)1 || shiftedEta == (double)1) {
			// This could be trivially handled as a special case if ever needed.
			result.success = false;
			return result;
		}

		// Get the non-normalized half vector.
		Vector3 tangentSpaceHalfVectorNonNormalizedMain;
		if (Frame::cosTheta(tangentSpaceMainWi) < (double)0) {
			tangentSpaceHalfVectorNonNormalizedMain = -(tangentSpaceMainWi * mainEta + tangentSpaceMainWo);
		}
		else {
			tangentSpaceHalfVectorNonNormalizedMain = -(tangentSpaceMainWi + tangentSpaceMainWo * mainEta);
		}

		// Get the normalized half vector.
		Vector3 tangentSpaceHalfVector = normalize(tangentSpaceHalfVectorNonNormalizedMain);

		// Refract to get the outgoing direction.
		Vector3 tangentSpaceShiftedWo = refract(tangentSpaceShiftedWi, tangentSpaceHalfVector, shiftedEta);

		// Refuse to shift between transmission and full internal reflection.
		// This shift would not be invertible: reflections always shift to other reflections.
		if (tangentSpaceShiftedWo.isZero()) {
			result.success = false;
			return result;
		}

		// Calculate the Jacobian.
		Vector3 tangentSpaceHalfVectorNonNormalizedShifted;
		if (Frame::cosTheta(tangentSpaceShiftedWi) < (double)0) {
			tangentSpaceHalfVectorNonNormalizedShifted = -(tangentSpaceShiftedWi * shiftedEta + tangentSpaceShiftedWo);
		}
		else {
			tangentSpaceHalfVectorNonNormalizedShifted = -(tangentSpaceShiftedWi + tangentSpaceShiftedWo * shiftedEta);
		}

		double hLengthSquared = tangentSpaceHalfVectorNonNormalizedShifted.lengthSquared() / (Epsilon + tangentSpaceHalfVectorNonNormalizedMain.lengthSquared());
		double WoDotH = abs(dot(tangentSpaceMainWo, tangentSpaceHalfVector)) / (Epsilon + abs(dot(tangentSpaceShiftedWo, tangentSpaceHalfVector)));

		// Output results.
		result.success = true;
		result.wo = tangentSpaceShiftedWo;
		result.jacobian = hLengthSquared * WoDotH;
	}
	else {
		// Reflection.
		Vector3 tangentSpaceHalfVector = normalize(tangentSpaceMainWi + tangentSpaceMainWo);
		Vector3 tangentSpaceShiftedWo = reflect(tangentSpaceShiftedWi, tangentSpaceHalfVector);

		double WoDotH = dot(tangentSpaceShiftedWo, tangentSpaceHalfVector) / dot(tangentSpaceMainWo, tangentSpaceHalfVector);
		double jacobian = abs(WoDotH);

		result.success = true;
		result.wo = tangentSpaceShiftedWo;
		result.jacobian = jacobian;
	}

	return result;
}

MTS_NAMESPACE_END