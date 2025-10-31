import OpenAI from "openai";
import fs from "fs/promises";
import dotenv from "dotenv";
dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const cliFile = process.argv[2];
const defaultSnippets = [
  "A rain-soaked detective hunts replicants in a neon city.",
  "Two friends road-trip across the desert and uncover a mystery.",
  "A young wizard discovers a hidden school of magic.",
  "A crew explores a derelict spaceship and finds a hostile lifeform."
];

const loadSnippets = async () => {
  if (!cliFile) return defaultSnippets;
  const file = await fs.readFile(cliFile, "utf-8");
  return file.split("\n").map(s => s.trim()).filter(Boolean);
};

const embedAll = async (snippets) => {
  const resp = await openai.embeddings.create({
    model: 'text-embedding-ada-002',
    input: snippets
  })

  const output = resp?.data?.map((row, i) => {
    return {
      id: i,
      text: snippets[i],
      vector: row.embedding
    }
  })

  await fs.writeFile('./embeddings.json', JSON.stringify(output, null, 2))
  console.log(`✅ Wrote ${output.length} embeddings to embeddings.json`);
  console.log(`   Vector dimension: ${output[0].vector.length}`);
}

const main = async () => {
  try {
    const snippets = await loadSnippets();
    if (snippets.length === 0) {
      console.error("No snippets found. Add lines to your file or edit defaultSnippets.");
      process.exit(1);
    }
    await embedAll(snippets);
  } catch (err) {
    console.error("Failed to generate embeddings:", err.response?.data ?? err.message);
    process.exit(1);
  }
}

main()