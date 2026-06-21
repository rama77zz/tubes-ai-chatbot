/**
 * RAGEngine - Mesin pencarian konteks lokal berbasis TF-IDF + Cosine Similarity.
 *
 * Dipanggil dari server-v2.js dengan kontrak:
 *   - buildVectorIndex(documents)        -> membangun index TF-IDF dari semua dokumen
 *   - retrieveContext(query, docs, topK) -> mengembalikan array context teratas yang RELEVAN
 *   - buildContextBlock(contextItems)    -> menyusun context items jadi satu blok teks untuk prompt
 *   - clearCache()                       -> reset index (dipanggil saat knowledge.json disimpan ulang)
 *
 * Desain penting:
 *   - retrieveContext TIDAK akan memaksa mengembalikan topK dokumen kalau skornya
 *     terlalu rendah / tidak relevan. Ini supaya server-v2.js bisa membedakan secara
 *     JUJUR antara "ada konteks relevan" vs "tidak ada sama sekali", lalu bisa langsung
 *     fallback tanpa bergantung 100% pada kepatuhan LLM terhadap system prompt.
 */

const STOPWORDS = new Set([
    'yang', 'untuk', 'pada', 'ke', 'di', 'dari', 'dan', 'atau', 'ini', 'itu',
    'dengan', 'adalah', 'akan', 'saya', 'kamu', 'anda', 'kami', 'kita', 'dia',
    'mereka', 'apa', 'apakah', 'bagaimana', 'gimana', 'kenapa', 'mengapa',
    'kapan', 'dimana', 'siapa', 'bisa', 'tolong', 'mau', 'ada', 'tidak',
    'juga', 'saja', 'sih', 'dong', 'kak', 'min', 'ya', 'kok', 'nih', 'jika',
    'kalau', 'bila', 'tentang', 'mengenai', 'buat', 'ikut', 'nanya', 'kah',
    'syarat', 'cara', 'aturan', 'ketentuan', 'panduan', 'biar', 'supaya'
]);

// Skor cosine similarity minimum agar sebuah dokumen dianggap "relevan".
// Di bawah ini dianggap noise / tidak nyambung -> dibuang dari hasil.
const MIN_RELEVANCE_SCORE = 0.05;

function normalizeText(text) {
    return (text || '')
        .toLowerCase()
        .normalize('NFKD')
        .replace(/[^a-z0-9\s]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

/**
 * Stemming ringan ala Bahasa Indonesia: melepas imbuhan umum (prefix/suffix)
 * supaya "dibuka", "membuka", "terbuka", "bukaan" dkk dianggap mirip dengan "buka".
 * Ini BUKAN stemmer linguistik lengkap (seperti Sastrawi) -- sengaja dibuat
 * sederhana dan aman (hanya strip pola yang sangat umum) supaya tidak butuh
 * dependency tambahan dan tidak salah potong kata pendek.
 */
function lightStem(word) {
    if (word.length <= 4) return word; // kata pendek, jangan diutak-atik (risiko salah potong)

    let w = word;

    // Suffix umum
    const suffixes = ['kan', 'an', 'nya', 'lah', 'kah', 'pun'];
    for (const suf of suffixes) {
        if (w.endsWith(suf) && w.length - suf.length >= 3) {
            w = w.slice(0, -suf.length);
            break;
        }
    }

    // Prefix umum (di-, me-, me-n, pe-, ter-, ber-)
    const prefixes = ['di', 'me', 'pe', 'ter', 'ber', 'se'];
    for (const pre of prefixes) {
        if (w.startsWith(pre) && w.length - pre.length >= 3) {
            w = w.slice(pre.length);
            break;
        }
    }

    return w.length >= 3 ? w : word; // fallback ke kata asli kalau hasil stem terlalu pendek
}

function tokenize(text) {
    return normalizeText(text)
        .split(' ')
        .filter(tok => tok.length > 1 && !STOPWORDS.has(tok))
        .map(lightStem);
}

class RAGEngine {
    constructor() {
        this.documents = [];       // dokumen mentah { pageContent, metadata }
        this.docTokens = [];       // token per dokumen (array of array)
        this.docVectors = [];      // vektor TF-IDF per dokumen (Map term -> weight)
        this.idf = new Map();      // inverse document frequency per term
        this.vocabulary = new Set();
        this.isBuilt = false;
    }

    /**
     * Reset seluruh index. Dipanggil saat knowledge.json / dataset berubah,
     * supaya index lama tidak dipakai lagi (mis. server-v2.js memanggil ini
     * di dalam saveKnowledge()).
     */
    clearCache() {
        this.documents = [];
        this.docTokens = [];
        this.docVectors = [];
        this.idf = new Map();
        this.vocabulary = new Set();
        this.isBuilt = false;
    }

    /**
     * Membangun index TF-IDF dari seluruh dokumen.
     * @param {Array<{pageContent: string, metadata: object}>} documents
     */
    buildVectorIndex(documents) {
        this.documents = Array.isArray(documents) ? documents : [];

        if (this.documents.length === 0) {
            console.warn('[RAGEngine] Tidak ada dokumen untuk dibangun indexnya.');
            this.isBuilt = false;
            return;
        }

        // 1. Tokenisasi semua dokumen
        this.docTokens = this.documents.map(doc => tokenize(doc.pageContent));

        // 2. Hitung Document Frequency (DF) per term
        const df = new Map();
        this.docTokens.forEach(tokens => {
            const uniqueTerms = new Set(tokens);
            uniqueTerms.forEach(term => {
                df.set(term, (df.get(term) || 0) + 1);
                this.vocabulary.add(term);
            });
        });

        // 3. Hitung IDF per term: log(N / df) dengan smoothing +1 agar tidak pernah 0/negatif
        const totalDocs = this.documents.length;
        this.idf = new Map();
        df.forEach((freq, term) => {
            this.idf.set(term, Math.log((totalDocs + 1) / (freq + 1)) + 1);
        });

        // 4. Hitung vektor TF-IDF per dokumen (dinormalisasi / unit vector)
        this.docVectors = this.docTokens.map(tokens => this._computeTfIdfVector(tokens));

        this.isBuilt = true;
        console.log(`[RAGEngine] Index dibangun: ${totalDocs} dokumen, ${this.vocabulary.size} term unik.`);
    }

    /**
     * Menghitung vektor TF-IDF ternormalisasi (unit length) dari sekumpulan token,
     * menggunakan idf yang sudah dihitung dari corpus (this.idf).
     */
    _computeTfIdfVector(tokens) {
        const tf = new Map();
        tokens.forEach(term => {
            tf.set(term, (tf.get(term) || 0) + 1);
        });

        const vector = new Map();
        let normSquared = 0;

        tf.forEach((count, term) => {
            const idfValue = this.idf.get(term);
            if (idfValue === undefined) return; // term tidak ada di corpus, abaikan
            const weight = count * idfValue;
            vector.set(term, weight);
            normSquared += weight * weight;
        });

        const norm = Math.sqrt(normSquared) || 1;
        vector.forEach((weight, term) => {
            vector.set(term, weight / norm);
        });

        return vector;
    }

    /**
     * Cosine similarity antara dua vektor (Map term -> weight), keduanya sudah unit-normalized.
     */
    _cosineSimilarity(vecA, vecB) {
        // Iterasi di vektor yang lebih kecil supaya lebih cepat
        const [small, big] = vecA.size <= vecB.size ? [vecA, vecB] : [vecB, vecA];
        let dot = 0;
        small.forEach((weight, term) => {
            if (big.has(term)) {
                dot += weight * big.get(term);
            }
        });
        return dot; // karena kedua vektor sudah unit-normalized, dot product = cosine similarity
    }

    /**
     * Mengambil topK dokumen paling relevan terhadap query.
     * Mengembalikan array KOSONG kalau memang tidak ada yang cukup relevan
     * (skor di bawah MIN_RELEVANCE_SCORE) -- ini disengaja, supaya server-v2.js
     * bisa langsung fallback tanpa bergantung pada kepatuhan LLM ke system prompt.
     *
     * @param {string} query
     * @param {Array} documents - dipakai untuk auto-build index kalau belum dibangun / berubah
     * @param {number} topK
     * @returns {Array<{pageContent: string, metadata: object, score: number}>}
     */
    retrieveContext(query, documents, topK = 6) {
        // Auto (re)build index kalau belum pernah dibangun, atau jumlah dokumen berubah
        if (!this.isBuilt || this.documents.length !== (documents ? documents.length : 0)) {
            this.buildVectorIndex(documents || []);
        }

        if (!this.isBuilt || this.documents.length === 0) {
            return [];
        }

        const queryTokens = tokenize(query);
        if (queryTokens.length === 0) {
            return [];
        }

        const queryVector = this._computeTfIdfVector(queryTokens);
        if (queryVector.size === 0) {
            // Semua kata di query tidak dikenal sama sekali oleh corpus (di luar vocabulary)
            return [];
        }

        const scored = this.documents.map((doc, idx) => {
            const score = this._cosineSimilarity(queryVector, this.docVectors[idx]);
            return { pageContent: doc.pageContent, metadata: doc.metadata, score };
        });

        scored.sort((a, b) => b.score - a.score);

        const relevant = scored.filter(item => item.score >= MIN_RELEVANCE_SCORE);

        if (relevant.length === 0) {
            console.log(`[RAGEngine] Tidak ada dokumen relevan untuk query: "${query}" (top score: ${scored[0]?.score.toFixed(4) || 0})`);
            return [];
        }

        return relevant.slice(0, topK);
    }

    /**
     * Menyusun context items menjadi satu blok teks siap pakai untuk system prompt.
     * @param {Array<{pageContent: string, metadata: object, score?: number}>} contextItems
     * @returns {string}
     */
    buildContextBlock(contextItems) {
        if (!contextItems || contextItems.length === 0) {
            return '';
        }

        return contextItems
            .map((item, idx) => {
                const topik = item.metadata?.topik || `Konteks ${idx + 1}`;
                const kategori = item.metadata?.kategori ? ` (${item.metadata.kategori})` : '';
                return `[${idx + 1}] ${topik}${kategori}:\n${item.pageContent}`;
            })
            .join('\n\n');
    }
}

module.exports = RAGEngine;
