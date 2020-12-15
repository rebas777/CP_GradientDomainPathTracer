#include "gdpt.h"

MTS_NAMESPACE_BEGIN

GDPTIntegrator::GDPTIntegrator(const Properties& props)
	: MonteCarloIntegrator(props)
{
	m_config.m_maxDepth = props.getInteger("maxDepth", -1);
	m_config.m_shiftThreshold = (double)props.getFloat("shiftThreshold", 0.001);
}

GDPTIntegrator::GDPTIntegrator(Stream* stream, InstanceManager* manager)
	: MonteCarloIntegrator(stream, manager)
{
	m_config = GDPTConfiguration(stream);
}

void GDPTIntegrator::serialize(Stream* stream, InstanceManager* manager) const
{
	MonteCarloIntegrator::serialize(stream, manager);
	stream->writeInt(m_maxDepth);
	m_config.serialize(stream);
}

Spectrum GDPTIntegrator::Li(const RayDifferential& r, RadianceQueryRecord& rRec) const {
	return m_color;
}

inline std::pair<double, double> GradientPathTracer::Weighting(const double rayPdf, const double lightSamplePdf, const double bsdfSamplePdf) {
	// Power heuristic weights for the following strategies: light sample from base, BSDF sample from base.
	// Combining sample the light source and BSDF with power heuristic to get better result
	double numerator = rayPdf * lightSamplePdf;
	double denominator = (rayPdf * rayPdf) * ((lightSamplePdf * lightSamplePdf) + (bsdfSamplePdf * bsdfSamplePdf));
	return std::make_pair(numerator, denominator);
}

/// Samples a direction according to the BSDF at the given ray position.
inline BSDFSampleResult sampleBSDF(RayState& rayState) {
	Intersection& its = rayState.rRec.its;
	RadianceQueryRecord& rRec = rayState.rRec;
	RayDifferential& ray = rayState.ray;

	// Note: If the base path's BSDF evaluation uses random numbers, it would be beneficial to use the same random numbers for the offset path's BSDF.
	//       This is not done currently.

	const BSDF* bsdf = its.getBSDF(ray);

	// Sample BSDF * cos(theta).
	BSDFSampleResult result = {
		BSDFSamplingRecord(its, rRec.sampler, ERadiance),
		Spectrum(),
		(double)0
	};

	Point2 sample = rRec.nextSample2D();
	result.weight = bsdf->sample(result.bRec, result.pdf, sample);

	// Variable result.pdf will be 0 if the BSDF sampler failed to produce a valid direction.
	// SAssert(result.pdf <= (Float)0 || fabs(result.bRec.wo.length() - 1.0) < 0.00001);
	return result;
}

void GradientPathTracer::evaluatePoint(RadianceQueryRecord& rRec, const Point2& samplePosition, const Point2& apertureSample, double timeSample, double differentialScaleFactor,
	Spectrum& out_very_direct, Spectrum& out_throughput, Spectrum* out_gradients, Spectrum* out_neighborThroughputs)
{
	// Initialize the base path.
	RayState mainRay;
	mainRay.throughput = m_sensor->sampleRayDifferential(mainRay.ray, samplePosition, apertureSample, timeSample);
	mainRay.ray.scaleDifferential(differentialScaleFactor);
	mainRay.rRec = rRec;
	mainRay.rRec.its = rRec.its;

	// Initialize the offset paths.
	RayState shiftedRays[4];

	static const Vector2 pixelShifts[4] = {
		Vector2(1.0f, 0.0f),
		Vector2(0.0f, 1.0f),
		Vector2(-1.0f, 0.0f),
		Vector2(0.0f, -1.0f)
	};

	for (int i = 0; i < 4; ++i) {
		shiftedRays[i].throughput = m_sensor->sampleRayDifferential(shiftedRays[i].ray, samplePosition + pixelShifts[i], apertureSample, timeSample);
		shiftedRays[i].ray.scaleDifferential(differentialScaleFactor);
		shiftedRays[i].rRec = rRec;
		shiftedRays[i].rRec.its = rRec.its;
	}

	// Evaluate the gradients. The actual algorithm happens here.
	Spectrum very_direct = Spectrum(0.0f);
	evaluate(mainRay, shiftedRays, 4, very_direct);

	// Output results.
	out_very_direct = very_direct;
	out_throughput = mainRay.radiance;

	for (int i = 0; i < 4; i++) {
		out_gradients[i] = shiftedRays[i].gradient;
		out_neighborThroughputs[i] = shiftedRays[i].radiance;
	}
}

void GradientPathTracer::evaluate(RayState& baseRay, RayState* offsetRays, int secondaryCount, Spectrum& out_veryDirect) {

	const Scene* scene = baseRay.rRec.scene;

	// Base path first intersection
	baseRay.rRec.rayIntersect(baseRay.ray);
	baseRay.ray.mint = Epsilon;

	// Offset rays second intersection
	for (int i = 0; i < secondaryCount; i++) {
		RayState& offsetRay = offsetRays[i];
		offsetRay.rRec.rayIntersect(offsetRay.ray);
		offsetRay.ray.mint = Epsilon;
	}

	if (!baseRay.rRec.its.isValid()) { // No valid intersection surface found
		// Add potential very direct light from the environment as gradients are not used for that.
		if (baseRay.rRec.type & RadianceQueryRecord::EEmittedRadiance) { // light source
			out_veryDirect += baseRay.throughput * scene->evalEnvironment(baseRay.ray);
		}
		return;
	}

	if (baseRay.rRec.its.isEmitter() && (baseRay.rRec.type & RadianceQueryRecord::EEmittedRadiance)) { // non environment light source
		out_veryDirect += baseRay.throughput * baseRay.rRec.its.Le(-baseRay.ray.d); // radiance emitted to direction d
	}

	// No offset path if offset ray hits nothing
	for (int i = 0; i < secondaryCount; i++) {
		RayState& offsetRay = offsetRays[i];
		if (!offsetRay.rRec.its.isValid()) {
			offsetRay.alive = false;
		}
	}

	baseRay.rRec.depth = 1;

	while (baseRay.rRec.depth < m_config->m_maxDepth) {

		bool lastSegment = (baseRay.rRec.depth + 1 == m_config->m_maxDepth);

		// Direct illumination sampling
		// Sample incoming radiance from lights (next event estimation).
		const BSDF* baseBSDF = baseRay.rRec.its.getBSDF(baseRay.ray);
		if ((baseRay.rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) // Direct (surface) radiance
			&& (baseBSDF->getType() & BSDF::ESmooth)) {
			// Only use direct illumination sampling when the surface's
			// BSDF has smooth (i.e. non-Dirac delta) component */
			DirectSamplingRecord dRec(baseRay.rRec.its);

			// Emission
			mitsuba::Point2 lightSample = baseRay.rRec.nextSample2D();
			// Re-wrote sampleEmitterDirect since the emittor with pdf 0 still could be sampled by light sampler [QUES]
			std::pair<Spectrum, bool> emitterSample = m_scene->sampleEmitterDirectVisible(dRec, lightSample); // Sample a position on an emitter
			Spectrum mainEmitterRadiance = emitterSample.first * dRec.pdf;
			bool emitterVisible = emitterSample.second;
			const Emitter* emitter = static_cast<const Emitter*>(dRec.object);

			BSDFSamplingRecord baseBSDFRecord(baseRay.rRec.its, baseRay.rRec.its.toLocal(dRec.d), ERadiance);
			Spectrum baseBSDFValue = baseBSDF->eval(baseBSDFRecord); // BSDF(wi, wo) * cos(theta)

			// Calculate the probability density of having generated the sampled path segment by BSDF sampling. Note that if the emitter is not visible, the probability density is zero.
			double baseBsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle && emitterVisible) ? baseBSDF->pdf(baseBSDFRecord) : 0;

			double mainDistanceSquared = (baseRay.rRec.its.p - dRec.p).lengthSquared();
			double mainOpposingCosine = dot(dRec.n, (baseRay.rRec.its.p - dRec.p)) / sqrt(mainDistanceSquared);

			std::pair<double, double> mainWeight = Weighting(baseRay.pdf, dRec.pdf, baseBsdfPdf);
			double mainWeightNumerator = mainWeight.first;
			double mainWeightDenominator = mainWeight.second;


			if (dot(baseRay.rRec.its.geoFrame.n, dRec.d) * Frame::cosTheta(baseBSDFRecord.wo) > 0) {
				for (int i = 0; i < secondaryCount; i++) {
					RayState& offsetRay = offsetRays[i];

					Spectrum mainContribution(double(0));
					Spectrum shiftedContribution(double(0));
					double weight = double(0);

					bool shiftSuccessful = offsetRay.alive;

					if (shiftSuccessful) {
						double shiftedBsdfPdf = 0;
						double shiftedDRecPdf = 0;
						double jacobian = 0;
						Spectrum shiftedBsdfValue = Spectrum((double)0.0);
						Spectrum shiftedEmitterRadiance = Spectrum((double)0.0);
						if (offsetRay.connection_status == RAY_CONNECTED) {
							// Follow the base path. All relevant vertices are shared. 
							shiftedBsdfPdf = baseBsdfPdf;
							shiftedDRecPdf = dRec.pdf;
							shiftedBsdfValue = baseBSDFValue;
							shiftedEmitterRadiance = mainEmitterRadiance;
							jacobian = (double)1;
						}
						else if (offsetRay.connection_status == RAY_RECENTLY_CONNECTED) {
							// Follow the base path. The current vertex is shared, but the incoming directions differ.
							Vector3 incomingDirection = normalize(offsetRay.rRec.its.p - baseRay.rRec.its.p);
							BSDFSamplingRecord bRec(baseRay.rRec.its, baseRay.rRec.its.toLocal(incomingDirection), baseRay.rRec.its.toLocal(dRec.d), ERadiance);

							// Sample the BSDF.
							shiftedBsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle && emitterVisible) ? baseBSDF->pdf(bRec) : 0; // The BSDF sampler can not sample occluded path segments.
							shiftedDRecPdf = dRec.pdf;
							shiftedBsdfValue = baseBSDF->eval(bRec);
							shiftedEmitterRadiance = mainEmitterRadiance;
							jacobian = (double)1;
						}
						else {
							// Reconnect to the sampled light vertex. No shared vertices.
							const BSDF* shiftedBSDF = offsetRay.rRec.its.getBSDF(offsetRay.ray);

							// This implementation uses light sampling only for the reconnect-shift.
							// When one of the BSDFs is very glossy, light sampling essentially reduces to a failed shift anyway.
							bool mainAtPointLight = (dRec.measure == EDiscrete);

							VertexType mainVertexType = getVertexType(baseRay.rRec.its.getBSDF(baseRay.ray), baseRay.rRec.its, *m_config, BSDF::ESmooth);
							VertexType shiftedVertexType = getVertexType(offsetRay.rRec.its.getBSDF(offsetRay.ray), offsetRay.rRec.its, *m_config, BSDF::ESmooth);

							if (mainAtPointLight || (mainVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE)) {
								// Get emitter radiance.
								DirectSamplingRecord shiftedDRec(offsetRay.rRec.its);
								std::pair<Spectrum, bool> emitterSample = m_scene->sampleEmitterDirectVisible(shiftedDRec, lightSample);
								bool shiftedEmitterVisible = emitterSample.second;
								shiftedEmitterRadiance = emitterSample.first * shiftedDRec.pdf;
								shiftedDRecPdf = shiftedDRec.pdf;

								// Sample the BSDF.
								double shiftedDistanceSquared = (dRec.p - offsetRay.rRec.its.p).lengthSquared();
								Vector emitterDirection = (dRec.p - offsetRay.rRec.its.p) / sqrt(shiftedDistanceSquared);
								double shiftedOpposingCosine = -dot(dRec.n, emitterDirection);

								BSDFSamplingRecord bRec(offsetRay.rRec.its, offsetRay.rRec.its.toLocal(emitterDirection), ERadiance);

								shiftedBsdfValue = shiftedBSDF->eval(bRec);
								shiftedBsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle && shiftedEmitterVisible) ? shiftedBSDF->pdf(bRec) : 0;
								jacobian = std::abs(shiftedOpposingCosine * mainDistanceSquared) / (Epsilon + std::abs(mainOpposingCosine * shiftedDistanceSquared));


							}
						}
						// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
						std::pair<double, double> offsetWeight = Weighting(jacobian * offsetRay.pdf, shiftedDRecPdf, shiftedBsdfPdf);
						double shiftedWeightDenominator = offsetWeight.second;
						weight = mainWeightNumerator / (Epsilon + shiftedWeightDenominator + mainWeightDenominator);

						mainContribution = baseRay.throughput * (baseBSDFValue * mainEmitterRadiance);
						shiftedContribution = jacobian * offsetRay.throughput * (shiftedBsdfValue * shiftedEmitterRadiance);
					}

					if (!shiftSuccessful) {
						// The offset path cannot be generated; Set offset PDF and offset throughput to zero. This is what remains.
						// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset. (Offset path has zero PDF)
						double shiftedWeightDenominator = double(0);
						weight = mainWeightNumerator / (Epsilon + mainWeightDenominator);

						mainContribution = baseRay.throughput * (baseBSDFValue * mainEmitterRadiance);
						shiftedContribution = Spectrum((double)0);
					}

					// Note: Using also the offset paths for the throughput estimate, like we do here, provides some advantage when a large reconstruction alpha is used,
					// but using only throughputs of the base paths doesn't usually lose by much.

					offsetRay.addGradient(shiftedContribution - mainContribution, weight);
				}
			}
		}

		// BSDF sampling and emitter hits
		BSDFSampleResult baseBsdfResult = sampleBSDF(baseRay);

		if (baseBsdfResult.pdf <= (double)0.0) {
			// Impossible base path.
			break;
		}

		const Vector baseWo = baseRay.rRec.its.toWorld(baseBsdfResult.bRec.wo); // out ray to world coordinates

		// The old intersection structure is still needed after main.rRec.its gets updated.
		Intersection previousBaseRayIntersection = baseRay.rRec.its;

		DirectSamplingRecord baseDRec(baseRay.rRec.its);
		const BSDF* baseBSDF = baseRay.rRec.its.getBSDF(baseRay.ray);

		// Trace a ray in the sampled direction.
		bool mainHitEmitter = false;
		Spectrum mainEmitterRadiance = Spectrum((double)0);

		// Update the vertex types.
		VertexType mainVertexType = getVertexType(baseBSDF, previousBaseRayIntersection, *m_config, baseBsdfResult.bRec.sampledType);
		VertexType mainNextVertexType;

		baseRay.ray = Ray(baseRay.rRec.its.p, baseWo, baseRay.ray.time); // update ray

		if (scene->rayIntersect(baseRay.ray, baseRay.rRec.its)) {
			// Intersected something - check if it was a luminaire.
			if (baseRay.rRec.its.isEmitter()) {
				mainEmitterRadiance = baseRay.rRec.its.Le(-baseRay.ray.d);
				baseDRec.setQuery(baseRay.ray, baseRay.rRec.its);
				mainHitEmitter = true;
			}
			// Update the vertex type.
			mainNextVertexType = getVertexType(baseRay.rRec.its.getBSDF(baseRay.ray), baseRay.rRec.its, *m_config, baseBsdfResult.bRec.sampledType);
		}
		else {
			// Intersected nothing -- perhaps there is an environment map?
			const Emitter* env = scene->getEnvironmentEmitter();

			if (env) {
				// Hit the environment map.
				mainEmitterRadiance = env->evalEnvironment(baseRay.ray);
				if (!env->fillDirectSamplingRecord(baseDRec, baseRay.ray))
					break;
				mainHitEmitter = true;

				// Handle environment connection as diffuse (that's ~infinitely far away).
				// Update the vertex type.
				mainNextVertexType = VERTEX_TYPE_DIFFUSE;
			}
			else {
				// Nothing to do anymore.
				break;
			}
		}

		// Continue the shift.
		double baseBsdfPdf = baseBsdfResult.pdf;
		double basePreviousPdf = baseRay.pdf;
		// Update ray
		baseRay.throughput *= baseBsdfResult.weight * baseBsdfResult.pdf;
		baseRay.pdf *= baseBsdfResult.pdf;
		baseRay.eta *= baseBsdfResult.bRec.eta;

		// Compute the probability density of generating base path's direction using the implemented direct illumination sampling technique.
		const double baseLumPdf = (mainHitEmitter && !(baseBsdfResult.bRec.sampledType & BSDF::EDelta)) ?
			scene->pdfEmitterDirect(baseDRec) : 0;

		// Power heuristic weights for the following strategies: light sample from base, BSDF sample from base.
		double mainWeightNumerator = basePreviousPdf * baseBsdfResult.pdf; //[QUES]
		double mainWeightDenominator = (basePreviousPdf * basePreviousPdf) * ((baseLumPdf * baseLumPdf) + (baseBsdfPdf * baseBsdfPdf));

		for (int i = 0; i < secondaryCount; ++i) {
			RayState& offsetRay = offsetRays[i];

			Spectrum shiftedEmitterRadiance(double(0));
			Spectrum mainContribution(double(0));
			Spectrum shiftedContribution(double(0));
			double weight = 0.0;

			bool postponedShiftEnd = false; // Kills the shift after evaluating the current radiance.

			if (offsetRay.alive) {
				// The offset path is still good, so it makes sense to continue its construction.
				double shiftedPreviousPdf = offsetRay.pdf;

				if (offsetRay.connection_status == RAY_CONNECTED) {
					// The offset path keeps following the base path.
					// As all relevant vertices are shared, we can just reuse the sampled values.
					Spectrum shiftedBsdfValue = baseBsdfResult.weight * baseBsdfResult.pdf;
					double shiftedBsdfPdf = baseBsdfPdf;
					double shiftedLumPdf = baseLumPdf;
					Spectrum shiftedEmitterRadiance = mainEmitterRadiance;

					// Update throughput and pdf.
					offsetRay.throughput *= shiftedBsdfValue;
					offsetRay.pdf *= shiftedBsdfPdf;

					// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
					double shiftedWeightDenominator = (shiftedPreviousPdf * shiftedPreviousPdf) * ((shiftedLumPdf * shiftedLumPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
					weight = mainWeightNumerator / (Epsilon + shiftedWeightDenominator + mainWeightDenominator);

					mainContribution = baseRay.throughput * mainEmitterRadiance;
					shiftedContribution = offsetRay.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
				}
				else if (offsetRay.connection_status == RAY_RECENTLY_CONNECTED) {
					// Recently connected - follow the base path but evaluate BSDF to the new direction.
					Vector3 incomingDirection = normalize(offsetRay.rRec.its.p - baseRay.ray.o);
					BSDFSamplingRecord bRec(previousBaseRayIntersection, previousBaseRayIntersection.toLocal(incomingDirection), previousBaseRayIntersection.toLocal(baseRay.ray.d), ERadiance);

					// Note: mainBSDF is the BSDF at previousMainIts, which is the current position of the offset path.

					EMeasure measure = (baseBsdfResult.bRec.sampledType & BSDF::EDelta) ? EDiscrete : ESolidAngle;

					Spectrum shiftedBsdfValue = baseBSDF->eval(bRec, measure);
					double shiftedBsdfPdf = baseBSDF->pdf(bRec, measure);

					double shiftedLumPdf = baseLumPdf;
					Spectrum shiftedEmitterRadiance = mainEmitterRadiance;

					// Update throughput and pdf.
					offsetRay.throughput *= shiftedBsdfValue;
					offsetRay.pdf *= shiftedBsdfPdf;

					offsetRay.connection_status = RAY_CONNECTED;

					// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
					double shiftedWeightDenominator = (shiftedPreviousPdf * shiftedPreviousPdf) * ((shiftedLumPdf * shiftedLumPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
					weight = mainWeightNumerator / (Epsilon + shiftedWeightDenominator + mainWeightDenominator);

					mainContribution = baseRay.throughput * mainEmitterRadiance;
					shiftedContribution = offsetRay.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
				}
				else {
					// Not connected - apply either reconnection or half-vector duplication shift.

					const BSDF* shiftedBSDF = offsetRay.rRec.its.getBSDF(offsetRay.ray);

					// Update the vertex type of the offset path.
					VertexType shiftedVertexType = getVertexType(shiftedBSDF, offsetRay.rRec.its, *m_config, baseBsdfResult.bRec.sampledType);

					// Reconnect when current offset vertex, current base vertex and next base certex are diffuse
					// Or next base vertex us a point light
					if (mainVertexType == VERTEX_TYPE_DIFFUSE && mainNextVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE) {
						// Use reconnection shift.

						// Optimization: Skip the last raycast and BSDF evaluation for the offset path when it won't contribute and isn't needed anymore.
						if (!lastSegment || mainHitEmitter || baseRay.rRec.its.hasSubsurface()) {
							ReconnectionShiftResult shiftResult;
							bool environmentConnection = false;

							if (baseRay.rRec.its.isValid()) {
								// This is an actual reconnection shift.
								shiftResult = reconnectShift(m_scene, baseRay.ray.o, baseRay.rRec.its.p, offsetRay.rRec.its.p, baseRay.rRec.its.geoFrame.n, baseRay.ray.time);
							}
							else {
								// This is a reconnection at infinity in environment direction.
								const Emitter* env = m_scene->getEnvironmentEmitter();

								environmentConnection = true;
								shiftResult = environmentShift(m_scene, baseRay.ray, offsetRay.rRec.its.p);
							}

							if (!shiftResult.success) {
								// Failed to construct the offset path.
								offsetRay.alive = false;
								goto shift_failed;
							}

							Vector3 incomingDirection = -offsetRay.ray.d;
							Vector3 outgoingDirection = shiftResult.wo;

							BSDFSamplingRecord bRec(offsetRay.rRec.its, offsetRay.rRec.its.toLocal(incomingDirection), offsetRay.rRec.its.toLocal(outgoingDirection), ERadiance);


							// Evaluate the BRDF to the new direction.
							Spectrum shiftedBsdfValue = shiftedBSDF->eval(bRec);
							double shiftedBsdfPdf = shiftedBSDF->pdf(bRec);

							// Update throughput and pdf.
							offsetRay.throughput *= shiftedBsdfValue * shiftResult.jacobian;
							offsetRay.pdf *= shiftedBsdfPdf * shiftResult.jacobian;

							offsetRay.connection_status = RAY_RECENTLY_CONNECTED;

							if (mainHitEmitter || baseRay.rRec.its.hasSubsurface()) {
								// Also the offset path hit the emitter, as visibility was checked at reconnectShift or environmentShift.

								// Evaluate radiance to this direction.
								Spectrum shiftedEmitterRadiance(double(0));
								double shiftedLumPdf = double(0);

								if (baseRay.rRec.its.isValid()) {
									// Hit an object.
									if (mainHitEmitter) {
										shiftedEmitterRadiance = baseRay.rRec.its.Le(-outgoingDirection);

										// Evaluate the light sampling PDF of the new segment.
										DirectSamplingRecord shiftedDRec;
										shiftedDRec.p = baseDRec.p;
										shiftedDRec.n = baseDRec.n;
										shiftedDRec.dist = (baseDRec.p - offsetRay.rRec.its.p).length();
										shiftedDRec.d = (baseDRec.p - offsetRay.rRec.its.p) / shiftedDRec.dist;
										shiftedDRec.ref = baseDRec.ref;
										shiftedDRec.refN = offsetRay.rRec.its.shFrame.n;
										shiftedDRec.object = baseDRec.object;

										shiftedLumPdf = scene->pdfEmitterDirect(shiftedDRec);
									}
								}
								else {
									// Hit the environment.
									shiftedEmitterRadiance = mainEmitterRadiance;
									shiftedLumPdf = baseLumPdf;
								}

								// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
								double shiftedWeightDenominator = (shiftedPreviousPdf * shiftedPreviousPdf) * ((shiftedLumPdf * shiftedLumPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
								weight = mainWeightNumerator / (Epsilon + shiftedWeightDenominator + mainWeightDenominator);

								mainContribution = baseRay.throughput * mainEmitterRadiance;
								shiftedContribution = offsetRay.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
							}
						}
						else {
							// Use half-vector duplication shift. These paths could not have been sampled by light sampling (by our decision).
							Vector3 tangentSpaceIncomingDirection = offsetRay.rRec.its.toLocal(-offsetRay.ray.d);
							Vector3 tangentSpaceOutgoingDirection;
							Spectrum shiftedEmitterRadiance(double(0));

							const BSDF* shiftedBSDF = offsetRay.rRec.its.getBSDF(offsetRay.ray);

							// Deny shifts between Dirac and non-Dirac BSDFs.
							bool bothDelta = (baseBsdfResult.bRec.sampledType & BSDF::EDelta) && (shiftedBSDF->getType() & BSDF::EDelta);
							bool bothSmooth = (baseBsdfResult.bRec.sampledType & BSDF::ESmooth) && (shiftedBSDF->getType() & BSDF::ESmooth);
							if (!(bothDelta || bothSmooth)) {
								offsetRay.alive = false;
								goto half_vector_shift_failed;
							}

							// Apply the local shift.
							HalfVectorShiftResult shiftResult = halfVectorShift(baseBsdfResult.bRec.wi, baseBsdfResult.bRec.wo, offsetRay.rRec.its.toLocal(-offsetRay.ray.d), baseBSDF->getEta(), shiftedBSDF->getEta());

							if (baseBsdfResult.bRec.sampledType & BSDF::EDelta) {
								// Dirac delta integral is a point evaluation - no Jacobian determinant!
								shiftResult.jacobian = double(1);
							}

							if (shiftResult.success) {
								// Invertible shift, success.
								offsetRay.throughput *= shiftResult.jacobian;
								offsetRay.pdf *= shiftResult.jacobian;
								tangentSpaceOutgoingDirection = shiftResult.wo;
							}
							else {
								// The shift is non-invertible so kill it.
								offsetRay.alive = false;
								goto half_vector_shift_failed;
							}

							Vector3 outgoingDirection = offsetRay.rRec.its.toWorld(tangentSpaceOutgoingDirection);

							// Update throughput and pdf.
							BSDFSamplingRecord bRec(offsetRay.rRec.its, tangentSpaceIncomingDirection, tangentSpaceOutgoingDirection, ERadiance);
							EMeasure measure = (baseBsdfResult.bRec.sampledType & BSDF::EDelta) ? EDiscrete : ESolidAngle;

							offsetRay.throughput *= shiftedBSDF->eval(bRec, measure);
							offsetRay.pdf *= shiftedBSDF->pdf(bRec, measure);

							if (offsetRay.pdf == double(0)) {
								// Offset path is invalid!
								offsetRay.alive = false;
								goto half_vector_shift_failed;
							}


							// Update the vertex type.
							VertexType shiftedVertexType = getVertexType(shiftedBSDF, offsetRay.rRec.its, *m_config, baseBsdfResult.bRec.sampledType);

							// Trace the next hit point.
							offsetRay.ray = Ray(offsetRay.rRec.its.p, outgoingDirection, baseRay.ray.time);

							if (!scene->rayIntersect(offsetRay.ray, offsetRay.rRec.its)) {
								// Hit nothing - Evaluate environment radiance.
								const Emitter* env = scene->getEnvironmentEmitter();
								if (!env) {
									// Since base paths that hit nothing are not shifted, we must be symmetric and kill shifts that hit nothing.
									offsetRay.alive = false;
									goto half_vector_shift_failed;
								}
								if (baseRay.rRec.its.isValid()) {
									// Deny shifts between env and non-env.
									offsetRay.alive = false;
									goto half_vector_shift_failed;
								}

								if (mainVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE) {
									// Environment reconnection shift would have been used for the reverse direction!
									offsetRay.alive = false;
									goto half_vector_shift_failed;
								}

								// The offset path is no longer valid after this path segment.
								shiftedEmitterRadiance = env->evalEnvironment(offsetRay.ray);
								postponedShiftEnd = true;
							}
							else {
								// Hit something.

								if (!baseRay.rRec.its.isValid()) {
									// Deny shifts between env and non-env.
									offsetRay.alive = false;
									goto half_vector_shift_failed;
								}

								VertexType shiftedNextVertexType = getVertexType(shiftedBSDF, offsetRay.rRec.its, *m_config, baseBsdfResult.bRec.sampledType);

								// Make sure that the reverse shift would use this same strategy!
								// ==============================================================

								if (mainVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE && shiftedNextVertexType == VERTEX_TYPE_DIFFUSE) {
									// Non-invertible shift: the reverse-shift would use another strategy!
									offsetRay.alive = false;
									goto half_vector_shift_failed;
								}

								if (offsetRay.rRec.its.isEmitter()) {
									// Hit emitter.
									shiftedEmitterRadiance = offsetRay.rRec.its.Le(-offsetRay.ray.d);
								}
							}


						half_vector_shift_failed:
							if (offsetRay.alive) {
								// Evaluate radiance difference using power heuristic between BSDF samples from base and offset paths.
								// Note: No MIS with light sampling since we don't use it for this connection type.
								weight = baseRay.pdf / (offsetRay.pdf * offsetRay.pdf + baseRay.pdf * baseRay.pdf);
								mainContribution = baseRay.throughput * mainEmitterRadiance;
								shiftedContribution = offsetRay.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
							}
							else {
								// Handle the failure without taking MIS with light sampling, as we decided not to use it in the half-vector-duplication case.
								// Could have used it, but so far there has been no need. It doesn't seem to be very useful.
								weight = double(1) / baseRay.pdf;
								mainContribution = baseRay.throughput * mainEmitterRadiance;
								shiftedContribution = Spectrum(double(0));

								// Disable the failure detection below since the failure was already handled.
								offsetRay.alive = true;
								postponedShiftEnd = true;

								// (TODO: Restructure into smaller functions and get rid of the gotos... Although this may mean having lots of small functions with a large number of parameters.)
							}
						}
					}

				}
			}

		shift_failed:
			if (!offsetRay.alive) {
				// The offset path cannot be generated; Set offset PDF and offset throughput to zero.
				weight = mainWeightNumerator / (Epsilon + mainWeightDenominator);
				mainContribution = baseRay.throughput * mainEmitterRadiance;
				shiftedContribution = Spectrum((double)0);
			}

			offsetRay.addGradient(shiftedContribution - mainContribution, weight);

			if (postponedShiftEnd) {
				offsetRay.alive = false;
			}
		}

		// Stop if the base path hit the environment.
		baseRay.rRec.type = RadianceQueryRecord::ERadianceNoEmission;
		if (!baseRay.rRec.its.isValid() || !(baseRay.rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance)) {
			break;
		}

		if (baseRay.rRec.depth++ >= 5) {
			/* Russian roulette: try to keep path weights equal to one,
			   while accounting for the solid angle compression at refractive
			   index boundaries. Stop with at least some probability to avoid
			   getting stuck (e.g. due to total internal reflection) */

			double q = std::min((double)(baseRay.throughput / baseRay.pdf).max() * baseRay.eta * baseRay.eta, (double)0.95f);
			if (baseRay.rRec.nextSample1D() >= q)
				break;

			baseRay.pdf *= q;
			for (int i = 0; i < secondaryCount; ++i) {
				RayState& shifted = offsetRays[i];
				shifted.pdf *= q;
			}
		}
	}
}

MTS_IMPLEMENT_CLASS_S(GDPTIntegrator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GDPTIntegrator, "Gradient domain path tracing integrator");
MTS_NAMESPACE_END

