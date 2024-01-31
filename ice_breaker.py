from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
    load_dotenv()

    print("Hello LangChain!")

    information = """
    Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, according to the Bloomberg Billionaires Index, and $254 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[5][6]

A member of the wealthy South African Musk family, Elon was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University. However, Musk dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.

In October 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight and satellite communications company. In 2004, he became an early investor in electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, Musk helped create SolarCity, a solar-energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. In 2022, he acquired Twitter for $44 billion, rebranding the service to X. In March 2023, he founded xAI, an artificial intelligence company.

Musk has expressed views that have made him a polarizing figure;[7] with particular criticism directed towards his statements of COVID-19 misinformation and antisemitic conspiracy theories.[7][8][9][10] His ownership of Twitter has been similarly controversial, being marked by the laying off of a large number of employees and an increase in hate speech and misinformation on the website. In 2018, the U.S. Securities and Exchange Commission (SEC) sued Musk over alleged false statements that he had secured funding for a private takeover of Tesla. To settle the case, Musk stepped down as the chairman of Tesla and paid a $20 million fine.

Early life and education
Childhood and family
Further information: Musk family
Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa's administrative capital.[11][12] He is of British and Pennsylvania Dutch ancestry.[13][14] His mother, Maye Musk (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.[15][16][17] His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, and property developer, who partly owned a Zambian emerald mine near Lake Tanganyika, as well as a rental lodge at the Timbavati Private Nature Reserve.[18][19][20][21] Musk has a younger brother, Kimbal, and a younger sister, Tosca.[17][22]

Musk's family was wealthy during his youth.[21] His father was elected to the Pretoria City Council as a representative of the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid.[11] His maternal grandfather, Joshua N. Haldeman, was an American-born Canadian who took his family on record-breaking journeys to Africa and Australia in a single-engine Bellanca airplane.[23][24][25][26] After his parents divorced in 1980, Musk chose to live primarily with his father.[13][18] Musk later regretted his decision and became estranged from his father.[27] He has a paternal half-sister and a half-brother.[23][28]

In one incident, after having called a boy whose father had committed suicide "stupid", Musk was severely beaten and thrown down concrete steps. His father derided Elon for his behavior and showed no sympathy for him despite his injuries.[29][30] He was also an enthusiastic reader of books, later attributing his success in part to having read Benjamin Franklin: An American Life, Lord of the Flies, the Foundation series, and The Hitchhiker's Guide to the Galaxy.[31][32] At age ten, he developed an interest in computing and video games, teaching himself how to program from the VIC-20 user manual.[33] At age twelve, Musk sold his BASIC-based game Blastar to PC and Office Technology magazine for approximately $500.[34][35]

Education
An ornate school building
Musk graduated from Pretoria Boys High School in South Africa
Musk attended Waterkloof House Preparatory School, Bryanston High School, and then Pretoria Boys High School, where he graduated.[36] Musk was a good but not exceptional student, earning a 61 in Afrikaans and a B on his senior math certification.[37] Musk applied for a Canadian passport through his Canadian-born mother,[38][39] knowing that it would be easier to immigrate to the United States this way.[40] While waiting for his application to be processed, he attended the University of Pretoria for five months.[41]

Musk arrived in Canada in June 1989 and lived with a second cousin in Saskatchewan for a year,[42] working odd jobs at a farm and lumber mill.[43] In 1990, he entered Queen's University in Kingston, Ontario.[44][45]

Two years later, he transferred to the University of Pennsylvania, an Ivy League university in Philadelphia, where he earned two degrees, a Bachelor of Arts in physics, and a Bachelor of Science degree in economics from the university's Wharton School.[46][47][48][49] Although Musk said he earned the degrees in 1995, the University of Pennsylvania maintains that they were awarded in 1997.[50] He reportedly hosted large, ticketed house parties to help pay for tuition, and wrote a business plan for an electronic book-scanning service similar to Google Books.[51]

In 1994, Musk held two internships in Silicon Valley: one at energy storage startup Pinnacle Research Institute, which investigated electrolytic ultracapacitors for energy storage, and another at Palo Alto–based startup Rocket Science Games.[52][53] In 1995, he was accepted to a PhD program in materials science at Stanford University.[54][55] However, Musk decided to join the Internet boom, dropping out two days after being accepted and applied for a job at Netscape, to which he reportedly never received a response.[56][38]
    """

    summary_template = """
    given the information {information} about a person I want to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    res = chain.invoke(input={"information": information})
    print(res)
