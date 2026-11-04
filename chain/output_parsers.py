"""Output parsers + multi-step pipeline: translate, then summarize.

A chat model's `invoke()` returns a Message object; StrOutputParser
converts that into a plain string, so it can feed directly into the next
step of a pipeline (a dict, another template, etc.) without reaching for
`.content` manually.

This file also shows how to compose two chains into a bigger pipeline:
`{"text": translate_chain}` runs `translate_chain` first and places its
string output under the "text" key, which is exactly the variable the
next PromptTemplate expects.

Docs: https://docs.langchain.com/oss/python/langchain/messages
Reference: https://reference.langchain.com/python/langchain-core/output_parsers/StrOutputParser
Notion ("Chains" - Aula 4): Criando Pipeline com Mais Etapas
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

text = """
A forma de se mudar para trabalhar depende de qual visto você usa. Para você, há dois caminhos principais. É bom seguir os dois ao mesmo tempo.

1. Transferência pela Playlist (visto 482, patrocinado pela empresa)
O antigo 482 foi totalmente substituído pelo Skills in Demand (SID), com exigências salariais e de conformidade mais rígidas. Desde julho, o piso do stream Core Skills subiu para AUD 79.423 e o do Specialist Skills para AUD 146.576. Um dev sênior pode chegar ao segundo piso, que tem prioridade de processamento. É o caminho mais rápido se a empresa topar. Depois ele leva à residência permanente pelo 186. 
noborderslawgroup
eiglaw

2. Visto por pontos, sem precisar de empregador (189/190/491)
O que você precisa:

Avaliação da ACS (Australian Computer Society). Como seus diplomas não são de TI, a ACS deve descontar alguns anos de experiência. Mesmo assim, com mais de 20 anos, você ainda fica com mais de 8 anos válidos.
Inglês: tirar PTE 79+ em todas as partes vale o máximo de pontos.
NAATI CCL: dá +5 pontos.
Exames de saúde e antecedentes criminais para a família toda.

⚠️ Ponto crítico: a idade. Aos 45 anos, os 15 pontos de idade caem para zero, e a ACS e o PTE levam meses. Isso te dá uma janela de cerca de 2 anos para receber o convite. Eu começaria pela ACS e pelo PTE já.

Contexto atual: o 189 está bem restrito, com convites concentrados em saúde e educação. Por isso, para TI, o 190 com nomeação estadual é mais realista. Além disso, profissões de saúde, ensino e construção ganharam prioridade de processamento. Como você é veterinário e está fazendo a complementação pedagógica, vale avaliar essas ocupações como plano B. Mas a revalidação de veterinário pela AVBC exige provas e é um processo longo. 
noborderslawgroup
propakistani

O passaporte italiano não ajuda muito na Austrália. Para o seu perfil, ele não dá acesso a um visto de trabalho útil.

Antes de pagar qualquer taxa, recomendo uma consulta com um agente de migração registrado (MARA), porque as regras mudaram várias vezes em 2026.
"""

translate_template = PromptTemplate(
    input_variables=["texto"],
    template = "Translate to english the following text:\n```{texto}```",
)

summarize_template = PromptTemplate(
    input_variables=["text"],
    template = "Summarize, using less than 10 words, the following text:\n```{text}```",
)

model = ChatAnthropic(model="claude-sonnet-4-6")

# Step 1: translate -> StrOutputParser gives back a plain string instead
# of a Message object, so it can be reused as plain text below.
translate_chain = translate_template | model | StrOutputParser()

# Step 2: wrap translate_chain's string output as {"text": ...} so it
# matches summarize_template's "text" placeholder, then summarize it.
summarize_chain = {"text": translate_chain} | summarize_template | model | StrOutputParser()

result = summarize_chain.invoke({"texto": text})
print(result)