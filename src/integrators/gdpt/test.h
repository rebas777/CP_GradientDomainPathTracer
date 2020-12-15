#pragma once

#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

class GDPTIntegrator : public MonteCarloIntegrator {

public:

	// Initialize the integrator with the specified properties
	GDPTIntegrator(const Properties& props);

	// Unserialize from a binary data stream
	GDPTIntegrator(Stream* stream, InstanceManager* manager);

	// Serialize to a binary data stream
	void serialize(Stream* stream, InstanceManager* manager) const;

	// Query for an unbiased estimate of the radiance along <tt>r</tt>
	Spectrum Li(const RayDifferential& r, RadianceQueryRecord& rRec) const {
		return m_color;
	}

private:

	Spectrum m_color;

	

};

MTS_IMPLEMENT_CLASS_S(GDPTIntegrator, false, MonteCarloIntegrator)

MTS_EXPORT_PLUGIN(GDPTIntegrator, "Gradient domain path tracing integrator");

MTS_NAMESPACE_END