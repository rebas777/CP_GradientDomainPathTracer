#include <mitsuba/render/imageblock.h>
#include <mitsuba/core/fresolver.h>


MTS_NAMESPACE_BEGIN

class GDPTWorkResult : public WorkResult {

	/* ==================================================================== */
	/*                             Work result                              */
	/* ==================================================================== */

	/**
	   Gradient-Domain path tracing needs its own WorkResult implementation,
	   since it needs to accumulate sampled data into multiple buffers simultaneously.
	*/
public:
	const size_t BUFFER_COUNT = 5;

	GDPTWorkResult(const ReconstructionFilter* filter,
		Vector2i blockSize = Vector2i(-1, -1), int extraBorder = 0);

	// Clear the contents of the work result
	void clear();

	/// Fill the work result with content acquired from a binary data stream
	virtual void load(Stream* stream);

	/// Serialize a work result to a binary data stream
	virtual void save(Stream* stream) const;

	/// Accumulate another work result into this one
	void put(const GDPTWorkResult* workResult);

	/// Add a sample to a buffer.
	inline void put(const Point2& sample, const Spectrum& spec, float alpha, float weight, int buffer) {
		double temp[SPECTRUM_SAMPLES + 2];
		for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
			temp[i] = spec[i];
		temp[SPECTRUM_SAMPLES] = 1.0f;
		temp[SPECTRUM_SAMPLES + 1] = weight;

		m_block[buffer]->put(sample, temp);
	}

	inline const ImageBlock* getImageBlock(int buffer = 0) const {
		return m_block[buffer].get();
	}

	inline void setSize(const Vector2i& size) {
		for (size_t i = 0; i < m_block.size(); ++i)
			m_block[i]->setSize(size);
	}

	inline const Point2i& getOffset() const {
		return m_block[0]->getOffset();
	}

	inline void setOffset(const Point2i& offset) {
		for (size_t i = 0; i < m_block.size(); ++i)
			m_block[i]->setOffset(offset);
	}

	/// Return a string representation
	std::string toString() const;

	MTS_DECLARE_CLASS()

protected:
	/// Virtual destructor
	virtual ~GDPTWorkResult();

protected:
	ref_vector<ImageBlock> m_block;
};


MTS_NAMESPACE_END