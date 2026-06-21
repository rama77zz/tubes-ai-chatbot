class RAGEngine {
    /**
     * Memecah teks menjadi token kata-kata bersih (huruf kecil alfanumerik)
     * @param {string} text 
     * @returns {Array<string>}
     */
    tokenize(text) {
        return text.toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ')
            .split(/\s+/)
            .filter(token => token.length > 1);
    }

    /**
     * Menyaring dan mengurutkan dokumen berdasarkan skor relevansi TF-IDF terhadap kueri
     * @param {string} query Kueri pencarian yang sudah diekspansi dari server
     * @param {Array} documents Array dokumen hasil keluaran dari dataset.js
     * @param {number} topK Jumlah dokumen teratas yang ingin diambil
     * @returns {Array} Potongan dokumen paling relevan untuk dikirim ke Groq AI
     */
    retrieveContext(query, documents, topK = 5) {
        const queryTokens = this.tokenize(query);
        if (queryTokens.length === 0 || documents.length === 0) return [];

        // 1. Hitung Document Frequency (DF) untuk setiap token di seluruh korpus
        const docCount = documents.length;
        const df = {};

        documents.forEach(doc => {
            const tokens = new Set(this.tokenize(doc.pageContent + " " + (doc.metadata.topik || "")));
            tokens.forEach(token => {
                df[token] = (df[token] || 0) + 1;
            });
        });

        // 2. Hitung skor kecocokan TF-IDF untuk setiap dokumen
        const scoredDocs = documents.map(doc => {
            const docTokens = this.tokenize(doc.pageContent + " " + (doc.metadata.topik || ""));
            
            // Hitung Term Frequency (TF) di dokumen ini
            const tf = {};
            docTokens.forEach(token => {
                tf[token] = (tf[token] || 0) + 1;
            });

            // Akumulasikan skor TF-IDF berdasarkan token kueri
            let score = 0;
            queryTokens.forEach(token => {
                if (tf[token] && df[token]) {
                    const termFreq = tf[token] / docTokens.length;
                    const invDocFreq = Math.log(docCount / df[token]) + 1;
                    
                    // Berikan bobot ekstra jika kata kunci COCOK pada judul TOPIK metadata
                    let weightMultiplier = 1.0;
                    if (doc.metadata.topik && doc.metadata.topik.toLowerCase().includes(token)) {
                        weightMultiplier = 2.5; // Judul topik diberi bobot lebih tinggi
                    }

                    score += (termFreq * invDocFreq) * weightMultiplier;
                }
            });

            return { doc, score };
        });

        // 3. Urutkan dari skor tertinggi dan saring yang memiliki skor > 0
        return scoredDocs
            .filter(item => item.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, topK)
            .map(item => item.doc);
    }
}

module.exports = RAGEngine;
