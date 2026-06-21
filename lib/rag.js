class RAGEngine {
    /**
     * Memecah teks menjadi token kata-kata bersih
     */
    tokenize(text) {
        return text.toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ')
            .split(/\s+/)
            .filter(token => token.length > 1);
    }

    /**
     * Menyaring dan mengurutkan dokumen berdasarkan skor relevansi TF-IDF terhadap kueri
     */
    retrieveContext(query, documents, topK = 5) {
        const queryTokens = this.tokenize(query);
        if (queryTokens.length === 0 || documents.length === 0) return [];

        const docCount = documents.length;
        const df = {};

        documents.forEach(doc => {
            const tokens = new Set(this.tokenize(doc.pageContent + " " + (doc.metadata.topik || "")));
            tokens.forEach(token => {
                df[token] = (df[token] || 0) + 1;
            });
        });

        const scoredDocs = documents.map(doc => {
            const docTokens = this.tokenize(doc.pageContent + " " + (doc.metadata.topik || ""));
            
            const tf = {};
            docTokens.forEach(token => {
                tf[token] = (tf[token] || 0) + 1;
            });

            let score = 0;
            queryTokens.forEach(token => {
                if (tf[token] && df[token]) {
                    const termFreq = tf[token] / docTokens.length;
                    const invDocFreq = Math.log(docCount / df[token]) + 1;
                    
                    let weightMultiplier = 1.0;
                    if (doc.metadata.topik && doc.metadata.topik.toLowerCase().includes(token)) {
                        weightMultiplier = 2.5;
                    }

                    score += (termFreq * invDocFreq) * weightMultiplier;
                }
            });

            return { doc, score };
        });

        return scoredDocs
            .filter(item => item.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, topK)
            .map(item => item.doc);
    }

    /**
     * Menggabungkan potongan dokumen menjadi satu teks blok untuk Groq AI
     */
    buildContextBlock(contextItems) {
        if (!contextItems || contextItems.length === 0) return "";
        return contextItems.map((item, index) => {
            return `[Aturan ${index + 1} - Topik: ${item.metadata.topik}]\n${item.pageContent}`;
        }).join("\n\n");
    }

    // =========================================================================
    // FUNGSI PENAMPUNG (MOCK FUNCTIONS) AGAR SERVER-V2.JS TIDAK CRASH 500
    // =========================================================================
    buildVectorIndex(documents) {
        // Karena TF-IDF kita dihitung secara dinamis, fungsi ini dikosongkan dengan aman
        return true;
    }

    clearCache() {
        // Pembersihan cache dikosongkan dengan aman
        return true;
    }
}

module.exports = RAGEngine;
