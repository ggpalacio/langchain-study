"""Summarization - "map_reduce" strategy, built by hand with LCEL.

Unlike "stuff" (summarize.py), which sends every chunk in a single
prompt, "map_reduce" scales to much longer documents in two phases:
  1. Map: summarize each chunk independently (one LLM call per chunk).
  2. Reduce: summarize those chunk-summaries together into one final
     summary.
This keeps each individual call within the model's context window, at
the cost of more LLM calls.

`load_summarize_chain(model, chain_type="map_reduce")` (see summarize.py)
gives you this same strategy out of the box. This script instead wires
the map and reduce phases manually with plain LCEL pieces (prompts,
RunnableLambda, `.map()`) to show what that helper is actually doing
under the hood.

Docs: https://docs.langchain.com/oss/python/langchain/short-term-memory
Reference: https://reference.langchain.com/python/langchain-core/runnables/base/Runnable
Text splitters: https://docs.langchain.com/oss/python/integrations/splitters/index
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
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
# (Documents) that fit the model's context window. chunk_size=50 is
# deliberately tiny here just to force several chunks out of this short
# poem; chunk_overlap keeps a bit of shared text between chunks so
# context isn't lost at the cut points.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, chunk_overlap=10
)

docs = splitter.create_documents([long_text])

model = ChatAnthropic(model="claude-sonnet-4-6")

# --- Map stage: summarize each chunk on its own ---
map_prompt = PromptTemplate.from_template(
    "Write a concise summary of the following text: {text}"
)
map_chain = map_prompt | model | StrOutputParser()

# Turn each Document into the dict map_prompt expects ({"text": ...}).
prepare_map_chain = RunnableLambda(lambda docs: [{"text": d.page_content} for d in docs])

# .map() takes a Runnable built for a single input and turns it into one
# that runs it once per item of a list, invoking the model once per
# chunk and collecting the results back into a list of summaries.
map_stage = prepare_map_chain | map_chain.map()

# --- Reduce stage: combine the chunk summaries into one ---
reduce_prompt = PromptTemplate.from_template(
    "Combine the following summaries into a unique concise summary: {summaries}"
)
reduce_chain = reduce_prompt | model | StrOutputParser()

# Join the list of chunk summaries into the single "summaries" string
# reduce_prompt's placeholder expects.
prepare_reduce_chain = RunnableLambda(lambda summaries: {"summaries": "\n".join(summaries)})
reduce_stage = prepare_reduce_chain | reduce_chain

# map_stage's list of summaries feeds straight into reduce_stage.
pipeline = map_stage | reduce_stage
result = pipeline.invoke(docs)
print(result)