class RAGEngine {
    constructor() {
        this.tfIdfCache = null;
    }

    /**
     * Membangun indeks peta kata (TF-IDF) secara lokal dan instan di memori
     */
    buildVectorIndex(documents) {
        if (!documents || documents.length === 0) return [];
        
        console.log(`[RAG Engine] Membangun indeks TF-IDF Lokal untuk ${documents.length} chunks...`);
        
        const docCount = documents.length;
        const dfMap = {};

        // 1. Hitung Document Frequency (DF) untuk setiap kata unik
        documents.forEach(doc => {
            const words = this._tokenize(doc.pageContent);
            const uniqueWords = new Set(words);
            uniqueWords.forEach(word => {
                dfMap[word] = (dfMap[word] || 0) + 1;
            });
        });

        // 2. Hitung Inverse Document Frequency (IDF)
        const idfMap = {};
        for (const [word, count] of Object.entries(dfMap)) {
            // Rumus IDF standar: log(Total Dokumen / Dokumen yang mengandung kata tersebut)
            idfMap[word] = Math.log(docCount / count) + 1;
        }

        // 3. Bangun Term Frequency (TF) untuk setiap dokumen
        this.tfIdfCache = documents.map((doc) => {
            const words = this._tokenize(doc.pageContent);
            const tfMap = {};
            words.forEach(word => {
                tfMap[word] = (tfMap[word] || 0) + 1;
            });

            // Normalisasi nilai TF berdasarkan panjang teks dokumen
            for (const word in tfMap) {
                tfMap[word] = tfMap[word] / words.length;
            }

            return {
                text: doc.pageContent,
                source: doc.metadata ? (doc.metadata.topik || doc.metadata.kategori) : "Pedoman Akademik",
                tf: tfMap,
                idf: idfMap
            };
        });

        console.log(`[RAG Engine] Indeks TF-IDF Lokal selesai dibuat secara instan!`);
        return this.tfIdfCache;
    }

    /**
     * Mengambil konteks dokumen terdekat menggunakan metode pencarian kata kunci lokal
     */
    retrieveContext(query, documents = [], topk = 3) {
        // Jika cache memori kosong dan dokumen eksternal masuk, bangun indeksnya terlebih dahulu
        if (!this.tfIdfCache) {
            if (!documents || documents.length === 0) return [];
            this.buildVectorIndex(documents);
        }

        const queryWords = this._tokenize(query);
        if (queryWords.length === 0) return [];

        // Hitung skor kemiripan teks berdasarkan bobot kata kunci
        const scored = this.tfIdfCache.map(doc => {
            let score = 0;
            queryWords.forEach(word => {
                if (doc.tf[word] && doc.idf[word]) {
                    // Skor ditambahkan berdasarkan bobot TF-IDF kata kunci yang cocok
                    score += doc.tf[word] * doc.idf[word];
                }
            });

            return {
                text: doc.text,
                source: doc.source,
                score: score
            };
        });

        // Urutkan dokumen dari yang paling banyak mengandung kata kunci relevan
        return scored
            .filter(item => item.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, topk);
    }

    buildContextBlock(contextItems) {
        if (!contextItems || !contextItems.length) return '';
        return contextItems
            .map((item, idx) => `[Konteks ${idx + 1}] Sumber: ${item.source}\n${item.text}`)
            .join('\n\n');
    }

    /**
     * Fungsi pembersih teks menjadi token kata dasar
     */
    _tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ') // Hapus tanda baca
            .split(/\s+/)
            .filter(word => word.length > 2); // Hanya ambil kata yang bermakna (> 2 huruf)
    }

    clearCache() {
        this.tfIdfCache = null;
    }
}

module.exports = RAGEngine;