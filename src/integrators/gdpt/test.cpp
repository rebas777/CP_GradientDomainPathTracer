#include "gdpt.h"

MTS_NAMESPACE_BEGIN

GDPTIntegrator::GDPTIntegrator(const Properties& props) : MonteCarloIntegrator(props) 
{
	Spectrum defaultColor;
	defaultColor.fromLinearRGB(0.2f, 0.5f, 0.2f);
	m_color = props.getSpectrum("color", defaultColor);
}

GDPTIntegrator::GDPTIntegrator(Stream* stream, InstanceManager* manager) : MonteCarloIntegrator(stream, manager)
{
	m_color = Spectrum(stream);
	// m_config = GradientPathTracerConfig(stream); [TODO]
}

void GDPTIntegrator::serialize(Stream* stream, InstanceManager* manager) const {
	MonteCarloIntegrator::serialize(stream, manager);
	m_color.serialize(stream);
	// m_config.serialize(stream); [TODO]
}



MTS_NAMESPACE_END