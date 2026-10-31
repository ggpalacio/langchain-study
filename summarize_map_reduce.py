"""Summarization - "map_reduce" strategy: summarize chunks, then combine.

Unlike "stuff" (summarize_stuff.py), which sends every chunk in a single
prompt, "map_reduce" scales to much longer documents in two phases:
  1. Map: summarize each chunk independently (one LLM call per chunk).
  2. Reduce: summarize those chunk-summaries together into one final
     summary.
This keeps each individual call within the model's context window, at
the cost of more LLM calls.

Docs: https://docs.langchain.com/oss/python/langchain/short-term-memory
Reference: https://reference.langchain.com/python/integrations/langchain_classic/chains/summarize
Text splitters: https://docs.langchain.com/oss/python/integrations/splitters/index
"""
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

long_text = """
Canadá

Há um país onde o inverno escreve
com tinta branca sobre o chão inteiro,
e o silêncio, de tão vasto e leve,
parece o mundo antes do primeiro.

Os bordos ardem quando chega outubro,
vermelho, ouro, cobre e açafrão,
como se a mata, num gesto rubro,
guardasse o fogo para o fim do verão.

Lagos espelham montanhas antigas,
o alce atravessa a névoa da manhã,
e as auroras, em danças amigas,
pintam de verde a noite que se vã.

Cidades de vidro à beira do frio,
Toronto, Montreal, Vancouver ao mar,
mil línguas cruzando o mesmo rio,
mil sotaques aprendendo a morar.

É terra de quem chega e recomeça,
de mãos que aquecem outras mãos geladas,
onde a distância nunca é promessa
de solidão, mas de novas estradas.

Canadá: neve, pinho e horizonte,
um norte inteiro que nos ensina
que todo frio pede uma fonte,
e todo inverno, uma lareira acesa.
"""

# RecursiveCharacterTextSplitter breaks a big text into smaller chunks
# (Documents) that fit the model's context window. chunk_size=100 is
# deliberately tiny here just to force several chunks out of this short
# poem; chunk_overlap keeps a bit of shared text between chunks so
# context isn't lost at the cut points.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=25
)

docs = splitter.create_documents([long_text])
# for doc in docs:
#     print(doc.page_content)
#     print("-"*100)

model = ChatAnthropic(model="claude-sonnet-4-6")

# "map_reduce" = summarize each chunk on its own (map), then summarize
# the collected chunk-summaries into a single result (reduce).
# verbose=True prints every prompt sent in both phases.
chain_summarize = load_summarize_chain(model, chain_type="map_reduce", verbose=True)
result = chain_summarize.invoke({"input_documents": docs})
print(result["output_text"])