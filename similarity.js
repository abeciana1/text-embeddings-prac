import OpenAI from "openai";
import Embeddings from "./embeddings.json" with { type: "json" };
import dotenv from "dotenv";
dotenv.config();

const query = process.argv.slice(2).join(" ");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const sendToOpenAI = async (query) => {
  const resp = await openai.embeddings.create({
    model: 'text-embedding-ada-002',
    input: query
  })

  const output = await resp?.data?.map((row, i) => {
    return {
      id: i,
      text: query,
      vector: row.embedding
    }
  })

  // console.log(`✅ Wrote ${output.length} embeddings to embeddings.json`);
  // console.log(`   Vector dimension: ${output[0].vector.length}`);
  // console.log("OpenAI response:", output);
  return output;
}

const cosineSimilarity = (vectorA, vectorB) => {
  let dotProduct = 0
  let magnitudeA = 0
  let magnitudeB = 0

  console.log("Calculating cosine similarity...")
  // console.log("Vector A:", vectorA)
  // console.log("Vector B:", vectorB)

  for (let index in vectorA) {
    dotProduct += vectorA[index] * vectorB[index]
    magnitudeA += vectorA[index] ** 2
    magnitudeB += vectorB[index] ** 2
  }

  let sqrtA = magnitudeA = Math.sqrt(magnitudeA)
  let sqrtB = magnitudeB = Math.sqrt(magnitudeB)

  let denominator = sqrtA * sqrtB
  if (denominator === 0) return 0

  return dotProduct / (denominator)
}

const main = async () => {
  let queryVector;
  if (Embeddings.length === 0) {
    console.error("No embeddings found. Run the embedding script first.");
    process.exit(1);
  }

  if (!query) {
    console.log("Please provide a search query.");
    queryVector = await sendToOpenAI("cyberpunk detective story")
  } else {
    console.log(`Searching for: ${query}`);
    queryVector = await sendToOpenAI(query)
  }

  const scoredResults = []

  for (let embed of Embeddings) {
    const score = cosineSimilarity(queryVector[0].vector, embed.vector)
    scoredResults.push({ query, score, related: embed.text })
  }

  const sortedScores = scoredResults.sort((a, b) => b.score - a.score)
  const top3 = sortedScores.slice(0, 3)
  for (let i = 0; i < top3.length; i++) {
    console.log(`${i + 1}. (${top3[i].score.toFixed(4)}) ${top3[i].related}`)
  }
}

main()