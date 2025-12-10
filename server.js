import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';

const app = express();
const port = 3000;

// 1. Configurações básicas
app.use(cors()); // Libera acesso para o Frontend
app.use(express.json());

// 2. Configura o Multer para salvar a imagem na memória RAM temporariamente
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// 3. Inicializa o Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// 4. Função auxiliar para converter o buffer do arquivo para o formato do Gemini
function fileToGenerativePart(buffer, mimeType) {
  return {
    inlineData: {
      data: buffer.toString("base64"),
      mimeType
    },
  };
}

// 5. Rota Principal (Onde o "antigravit" recebe a foto)
app.post('/api/analisar-rosto', upload.single('foto'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Nenhuma imagem enviada.' });
    }

    // Configura o modelo (Flash é rápido para isso)
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    // Prepara a imagem
    const imagePart = fileToGenerativePart(req.file.buffer, req.file.mimetype);

    // O Prompt Mágico (Focado em JSON para facilitar seu frontend)
    const prompt = `
      Atue como um especialista em visagismo e estética facial. Analise esta imagem.
      O objetivo é "Looksmaxxing" (melhoria estética).
      
      Por favor, retorne APENAS um objeto JSON (sem crase, sem markdown) com a seguinte estrutura:
      {
        "simetria": {
          "nota": "0 a 10",
          "analise": "Descrição breve da simetria dos olhos, sobrancelhas e mandíbula."
        },
        "qualidade_pele": {
          "nota": "0 a 10",
          "analise": "Descrição de textura, acne ou manchas visíveis."
        },
        "formato_rosto": "Oval, Quadrado, Diamante, etc.",
        "pontos_fortes": ["ponto 1", "ponto 2"],
        "sugestoes_melhoria": [
          "Dica prática 1 (ex: estilo de barba ou cabelo)",
          "Dica prática 2 (ex: skincare)",
          "Dica prática 3 (ex: exercícios ou postura)"
        ]
      }
    `;

    // Envia para o Gemini
    const result = await model.generateContent([prompt, imagePart]);
    const response = await result.response;
    const text = response.text();
    
    console.log("Gemini Raw Response:", text); // Debug log

    // Robust JSON extraction
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error("Não foi possível encontrar um JSON válido na resposta da IA.");
    }
    
    const jsonString = jsonMatch[0];
    const data = JSON.parse(jsonString);

    // Devolve para o seu Frontend
    res.json(data);

  } catch (error) {
    console.error("Erro na análise:", error);
    res.status(500).json({ error: 'Erro ao processar imagem com IA. Verifique os logs do servidor.' });
  }
});

app.listen(port, () => {
  console.log(`Servidor rodando em http://localhost:${port}`);
});
