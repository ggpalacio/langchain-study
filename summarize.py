"""Summarization - "stuff" strategy: cram every chunk into one prompt.

`load_summarize_chain` builds a ready-made chain for summarizing a list
of Documents. The "stuff" chain_type takes the simplest approach: it
concatenates ("stuffs") every chunk into a single prompt and asks the
model to summarize it in one LLM call.

This only works while everything still fits in the model's context
window - for longer texts than the model can take in one shot, see the
"map_reduce" strategy in summarize_map_reduce.py.

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

# "stuff" = put all chunks together into one prompt and summarize in a
# single call. verbose=True prints the prompt(s) the chain actually sends.
#
# For texts too long to fit in one prompt, load_summarize_chain also
# accepts chain_type="map_reduce" as a drop-in replacement here: it
# summarizes each chunk separately, then combines those summaries into
# one. See summarize_map_reduce.py for that same strategy built by hand
# with LCEL instead of this one-line helper.
#
# load_summarize_chain also accepts a `prompt` kwarg (a PromptTemplate
# with a "text" variable, matching document_variable_name) to replace
# its default summarization instructions - e.g.
# prompt=PromptTemplate.from_template("Summarize in Portuguese:\n{text}")
# - without having to touch any other part of the chain.
chain_summarize = load_summarize_chain(model, chain_type="stuff", verbose=True)
result = chain_summarize.invoke({"input_documents": docs})
print(result["output_text"])