Abstract

In this paper we survey transformer architecture to understand the working of Large language models and the source of AI hallucination. The paper then dives into the recent development in AI agents. Agents mitigate hallucination by adding external knowledge and provides profiling, memory, planning and actions capabilities to large language models. There are multiple agent tools that assist in a variety of tasks. Due to the variety of agents and the wide range of tasks a language model can perform,  agent section lack cohesiveness. The paper suggest using the research triangulation to find the best agents for tasks by comparing multiple agent outputs. This system uses the principles of research triangulation and aims to reduce AI hallucination.

Keywords

Large language models, AI Hallucination, LLM Agents.



Introduction

Large language models can recognize and generate text in real-time. LLMs can perform a variety of tasks but are limited with the knowledge within its parameters.

The limited knowledge is a source of hallucination. AI agent use external sources though the prompt window of a large language model expanding the capabilities and knowledge of LLMs though  profiling, memory, planning and action models. Each module has multiple developments and use various architectures to complete tasks. A single tasks can be executed using different agents. The information model adopts a system where the most effective agent is selected to efficiently handle a task .

In this paper we propose the information model that compares multiple agent outputs and finds the best agent for a task. The model provides weights to the agents according to their performance. The information model uses the principles of research triangulation to find the best agent for a task. Research triangulation is a method used to increase the credibility and validity of research findings by comparing results from multiple data sources, Investigators and methods. Similarly, the information model compares different methods of agents and compares different data sources to draw conclusions.



Literature review

Transformer

Transformer is a neural network capable of unsupervised learning. As shown in figure 1, the transformer consists of encoder and decoder stacks. Words embedded in the input embedding of a transformer are called tokens. The encoder understands the meaning of a token and the relationship between tokens. The encoder updates the meaning of the sentence by adjusting its weights. The decoder understands the meaning of a prompt by comparing the prompt with the meaning stored in the encoder. By comparing the weights with other tokens, the model decides how similar or dissimilar a word is to each other. The model updates the weights of each token multiple times, enhancing the meaning of each token. With the newly generated embedding the model determines the probability of the next token. A transformer-based neural network allows token embedding in billions of parameters and can read large amounts of data while generating text. (Vaswani et al., 2017).  Encoder models like BERT are great at understanding language, while decoder models like GPT4 are great at text generation.



Encoder

Encoder has multiple layers stacked on each other that determine the weights of tokens. The encoder places token in a multi parametric matrix where each parameter holds meaning and the position captures essential information about the input. Encoders help a machine learning model understand human language. (Vaswani et al., 2017)

Figure 1 - The Transformer architecture(Vaswani et al., 2017)



Input embedding

Input embedding is to tokenise words into numbers called Input ID. Each ID is then embedded into a high dimensional vector representation. The matrix representation allows the model to capture semantic similarities between words. (Vaswani et al., 2017)

Positional Encoding

Positional Encoding is to position each token embedding in a sequential order through which it understands the meaning of a token. Each token is associated with a vector where the meaning of the token is encoded in the position. Tokens with coordinates closer together have similar meanings (Vaswani et al., 2017). For example, a dimension that contain the information of the size of an object would place the word 'ant' closer to a 'bee' and an 'elephant' closer to a 'giraffe' in the same dimension while the smaller objects would sequentially scale up to the larger objects.

The tokens have tunable weights, these weights/tunable parameters determine the point where the word is placed. With extensive training using datasets of books, articles and websites, the model arranges the weights of the tokens. The model recognizes each word and places similar words closer to each other. The machine progressively improves its understanding of language by adjusting the weights of the tokens in a matrix. The model keeps adjusting the location of the vector according to the use of a token in the training datasets. Adjusting the position of the vector changes the weights of the token that finally changes how the computer understands the meaning of a word. Hence direction and position hold meaning, the position of the words in a matrix is how a machine understands human language. (Vaswani et al., 2017)

The model stores multiple meanings of a token at the same point in space. The tokens do not get duplicated in the matrix. It can store multiple meanings of a single token due to the embeddings in high dimensions. Each dimension stores a type of information. It could be gender-based information, singular or plural, or living and non-living information. (Vaswani et al., 2017)

The Scaled Dot-product

Figure 2 - Scaled dot product attention(Vaswani et al., 2017)

As shown in Figure 2, The Scaled Dot-product makes the model understand the context of the combination of words by paying attention to the relationship of a word in context to the sentence. For example, a 'financial Bank' is different from a 'river bank'.  The model can understand the context by the dot product of the Query, Key and Value matrix. The query matrix consists of all the embedding of an input text. Producing a query vector for each token. The Key matrix, similar to the query matrix, also consists of all the embedding of the input text. The query key dot product compares all the words in the queries with the keys. When the key and the query vector closely align in the matrix the words are highly related and if they do not align, the words are not related. The query key matrix contains weights that denote the relation of all the words in the input text. The value matrix updates the weights of the embeddings allowing words that are highly related to pass information to each other. The weights assigned in the value matrix determine what should be added to a vector that reflects the meaning of another vector. The value matrix assigns weights to each query vector, and adds the weights to each Key vector, this updates the weights of an embedding with the values of another embedding. The updated values encapsulate the meaning of two embeddings in a single position for all the words that are relevant to adjusting the meaning of another word. The attention pattern matrix combines all three matrices, The output contains the combined meaning of multiple relevant words. (Vaswani et al., 2017)

Multi-head attention



Figure 3 – Multi-head Attention (Vaswani et al., 2017)

As shown in figure 3, The multi-head attention contains multiple sets of Scaled Dot-product attention blocks. The multiple heads of Query, Key and Value matrices enable the machine to understand different contexts of the same sentence. The Concat layer adds the values of the multiple scaled dot products together providing a rich meaning of the context. (Vaswani et al., 2017)

Add & Norm

The original weights of the embeddings before the attention layer are sent to the add and norm layer, enabling the model to retain their original meanings. The add layers add the values of the multiheaded attention layer with the original weights and the normalization calculates the standard deviation and the mean of the add layer and updates the weights of the embeddings with the normalised weights. (Vaswani et al., 2017)

Decoder

Decoder comprises of layers that compares encoder embeddings with output embeddings to make the model understand the context of the input text. Comparing the output embeddings with the encoder embeddings lets the model understand the context of the input embeddings so it can determine the probability of the next token. (Vaswani et al., 2017)

Output embedding

The output embedding is to tokenise works into numbers called ID. Each ID is then embedded into the embedding matrix. Similar to the input embedding. The output embedding contains the positional encoding of input tokens. (Vaswani et al., 2017)

Positional Encoding

The positional encoder is to position each output token in a matrix through which it understands the meaning of each word. The positional encoder converts tokens to embedding vectors that contain positional information of the tokens. The position of a vector holds meaning and can be understood relative to the position of another word. (Vaswani et al., 2017)

Masked multi-head attention

The masked multi-head attention block compares the query, key and value matrices of the output embeddings giving context to the sentence. The decoder generates new words. Each time a novel word is generated, this layer adds the newly generated word in the output embedding, updating the weights of the newly generated word with the context of the sentence. During the training phase the correct output is masked, the model generates a word and unmasks the answer to gain feedback. If the masked word and generated word match, the output is sent to the add and norm layer, if the words does not match the model is forced to choose another word. (Vaswani et al., 2017)

Add and norm layer

The add and norm layer converts Masked multiread attention layer to the value matrix. (Vaswani et al., 2017)

Multi-head attention layer

The multi-head attention layer uses the token embeddings from the encoder in the query and key matrix and uses the newly generated word embedding in the value matrix. Comparing the weights of the query key dot product with the value matrix makes the model understand the context of the input text in relation to the encoders understand. Attention block handles understanding context and meaning of the generated sentence. The multi-head attention updates weights of embeddings and incorporate the meaning of the whole sentence in a single point of the matrix. multiheaded attention understand the context of the paragraph helping the model chose the next plausible word. (Vaswani et al., 2017)

Linear

The Linear determines the probability of the next token with a probability score. It maps the positions output of multiheaded attention layer in the vocabulary matrix. The liner layer determines the vocabulary size. The probability decides how likely a word is to be chosen as an output. Temperature is a value that can be set by the user. The temperature decides the probability distribution of the selected words. A lower temperature would select words with the highest weights while the high temperature would allow for lower weights to be selected. (Vaswani et al., 2017)

SoftMax

SoftMax converts the weights in a way that rows of the matrix add up to 1. SoftMax makes sure that the words that come after do not interact with the words that come before. (Vaswani et al., 2017)

Generative AI

With the help of the transformer architecture a computer is able to generate text to text, text to image,  text to code, text to video, text to speech and text to music. Generative AI models learn the patterns through input training data and then generate new data with similar characteristics. There are multiple types of generative AI, in this paper we will focus on large language models.

Large language models

Large language models can recognize and generate text in real-time. LLMs are trained on large datasets. The data is carefully placed in billions of parameters that enable LLMs to understand patterns of language. LLMs can recognize patterns through the transformer architecture.

Multimodal large language models

The scale of large language models has grown recently at the beginning large language models preform text-to-text tasks, now LLMs are capable of handling other generative tasks like text to image, text to video. This capability is known as Multimodal large language models. Multimodality is the application of multiple literacies within one medium.



What is AI Hallucination

AI Hallucination is when generated content produced by LLM are nonsensical or unfaithful to the provided source content. Hallucinations are seemingly plausible generated information but are incoherent or non-factual. Hallucination can be categorised into various types from various sources. (Zhang et al., 2023)

Source of LLM Hallucination

LLM hallucination can occur due to data, training or during interface there could be lack of knowledge in the model dataset, misalignment during training or there could be an issue with the generation strategy which can lead to hallucination.

Hallucination from Data

When a model is trained on data with source reference divergence or lack of relevant knowledge the model generates text that is not faithful to the source. (Ji et al., 2022)

Lack of relevant knowledge

LLMs have a vast amount of knowledge stored in their parameters. If the LLM internalizes false knowledge during training, it will generate an incorrect response. LLMs may misinterpret positionally close associations as factual knowledge. (Zhang et al., 2023)

Source Reference Divergence

The source reference divergence is when the model learns to generate text that is not grounded to a given source. This could happen due to Heuristic data collection or Innate Divergence. Heuristic is when the information of the source mismatches with the information of the target ground-truth reference. The model generates text that is not based on reasoning, but it chooses a word based on how frequently a word is used in the training dataset. Another issue occurs when duplicates from the datasets are not filtered out. Innate Divergence is when there is no factual knowledge alignment between source input text and target reference. However, for the models that value diversity, innate divergence helps LLMs improve engagingness by diversifying dialogue generation. Innate divergence led to extrinsic hallucination. (Ji et al., 2022)

Hallucination from Training & Interference

Hallucination can occur even when there is little divergence during training due to the training and modelling choice of neural models (Ji et al., 2022)

Overestimating their own capabilities

LLMs with large parameter count are unable to determine the correctness of a response with certainty during self-evaluation. LLMs generate correct and incorrect answers with equal confidence. The LLM may generate incorrect response with high confidence. The high confidence misleads LLMs to fabricate answers with certainty. (Zhang et al., 2023)

Problematic alignment process

During pretraining, the model receives training on curated instruction to align their responses to human preferences. However, if the LLM does not have adequate knowledge from the pretraining phase it misaligns the tokens causing LLMs to hallucinate. (Zhang et al., 2023)

Generation strategy

The generation strategy is when the model overcommits to their early mistake, even when they recognize that is it incorrect. The LLMs do not recover from their error. This is called hallucination snowballing. The randomness from a higher temperature may lead to hallucination. (Zhang et al., 2023)

Imperfect representation learning

Imperfect representation learning occurs when encoders learn wrong correlation between different parts of training data, the encoders attain defected comprehension ability which could influence the model to generate erroneous responses. (Ji et al., 2022)

Erroneous decoding

Erroneous decoding can occur due to two reasons. First, decoder can attend to the wrong parts of the encoder source, these incorrect associations result in fact mix ups in two similar entities. Second, the design of the decoder could contribute to hallucination if the model deliberately adds randomness by setting a higher temperature, it increases the chances of unexpected text generation. (Ji et al., 2022).

Exposure Bias

During the training phase the decoder is trained in predicting the next token condition on the ground-truth. However, During Interface time, the model generates tokens according to the historical sequences previously generated by itself. This discrepancy leads to errored results when the target sequence gets longer. (Ji et al., 2022)

Parametric Knowledge bias

Models with large corpus memorize their knowledge in its parameters through positional encoding. This knowledge improves the performance of a task. The model that favors generating output with the prometric knowledge than the input of the users can result in AI hallucinations. (Ji et al., 2022)



Types of hallucination

There could be multiple kinds of hallucination ranging from small factual mistakes or fabricated contact, There are 5 categories of AI hallucination ntrinsic hallucination, xtrinsic hallucination, nput conflict hallucination, ontext conflict hallucination or act conflict hallucination. In this section we will be going through each type.  (Zhang et al., 2023)

Intrinsic hallucination

Intrinsic hallucination is when the generated output contradicts the source content. " (Ji et al., 2022) For example, model predicts "the current president of United States is George Bush instead of Joe Biden.

Extrinsic hallucination

Extrinsic hallucination is where the generated output cannot be supported or contradicted by the source. The LLM adds unverified information. For example, "India has started clinical trials for a new COVID-19 vaccine" this statement cannot be supported as it is not mentioned in the source. Although this kind of hallucination can be helpful in some cases as it recalls additional background knowledge and improves informativeness of generated text in some cases. (Ji et al., 2022)

Input conflict hallucination

Input conflict hallucination is where LLMs generate content that deviates from the source input provided by the user. It occurs due to the contradiction between the user input and the predicted word. The LLM may misunderstand the user's interests, paying attention to another part of input which causes the input conflict hallucination. (Zhang et al., 2023)

Context conflict hallucination

Context conflict hallucination is when LLMs generate self-conflicting content from the previously generated information. This occurs when the LLM lose track of the context or when it fails to mention consistency throughout the conversation. It occurs due to the limitations of the context window  (Zhang et al., 2023)

Fact conflict hallucination

Fact conflict hallucination is when LLMs generate content that is not faithful or is non-factual compared to the established word knowledge. The fact conflict hallucination can be introduced in multiple stages.(Zhang et al., 2023)

Detecting fact conflict hallucination

There are multiple automatic evaluation methods, FactSore and FactCHD both use external knowledge to detect fact conflict hallucination.



FactScore

Figure 4 - FactScore (Min et al., 2023)

FactScore adopts a perspective that trustworthiness of a sentences should depend on reliable knowledge source. As shown in figure 4, FactScore checks if information is supported by a large text corpus. Once a text is generated by an LLM, it breaks the texts into smaller chunks and verifies each fact with an external source like Wikipedia. It then gives a score to each chunk of text. As long text consists of many pieces of information that can each be true or false. The score determines the accuracy of the generated information. (Min et al., 2023)

FactCHD

Figure 5 - Factuality pattern overview(Chen et al., 2024)

FactCHD is a large-scale evaluation benchmark tailored for LLMs. The first step is Task formulation where the model detects fact conflict hallucination. The model labels LLM generated responses as factual or non-factual by checking them with external knowledge. As shown in figure 5, Factuality patterns aim to explore patterns of factual errors, it consists of vanilla pattern, multi-hops pattern, comparison pattern and set-operation patters. The Vanilla patters verify generated information with verified established sources, if a generated statement requires connecting multiple pieces of facts, the multi-hop patterns verify generated information from multiple sources. The comparison patters evaluate and compares different facts. The set operation patterns analyse relationship between different facts, manipulates multiple elements and combines them to form a statement. Based on the factually patterns a Query response is generated. The query responses provide a rich structured knowledge for computational reasoning that anchors factual data.

The FactCHD paper also introduces Truth Triangulator that verifies knowledge by cross referencing multiple independent perspectives using the fact verdict manager. As shown in figure 6, The truth triangulator improves detecting fact conflict hallucination in uncontrolled settings. Fact verdict manager comprises of two elements, Truth seeker that looks for information form the external information sources and Truth guardian that relies on the internal knowledge. The fact verdict manager compares the knowledge of the truth guardian with the truth seeker to collect evidence from multiple viewpoints which enhances the reliability and accuracy of the conclusion. (Chen et al., 2024)

Figure 6 – Truth Triangulator overview(Chen et al., 2024)



Mitigating fact conflict hallucination

Fact conflict hallucination can be mitigated during pre-training, during fine tuning, reinforcement learning or can be mitigated using external knowledge. This section focusses on mitigation from external sources. (Zhang et al., 2023)

Mitigation during Pre-training

Mitigation during pre-training involves manually eliminating noisy data from the corpus or employ tools for data selection and filtering. Noisy data could be misinformation in the corpus that may corrupt the parametric knowledge of LLMs. (Zhang et al., 2023)

Mitigation during Supervised fine training

It is where the pretrained data is further trained on specific labelled tasks and the model adjusts its parameters according to the tasks. These labels demonstrate the desired output guiding and fine tuning the model. The labellers update the text as 'supported' or 'non supported' giving feedback to the model. (Ouyang et al., 2022)

Mitigation during Reinforced learning

As shown in figure 7, Mitigation during reinforced from human feedback uses Supervised fine training as the first step, then asks the large language model to produce multiple outputs as options. The labellers train the reward model by ranking model generated outputs from best to worst. This optimizes according to human preferences. (Ouyang et al., 2022)

Figure 7 - Reinforcement learning overview(Ouyang et al., 2022)

Mitigation from External knowledge

External knowledge can be accessed through retrieval systems, like Retrieval-Augmented Generation system. (Zhang et al., 2023)

RAG system

Figure 8 – Rag system Overview (Lewis et al., 2021)



A RAG (Retrieval augmented generation) allows transformer model to retrieve information from external unstructured or non-parametric knowledge base. As shown in figure 8, The RAG system consists of a query encoder, a retriever, and a document index. The relevant documents are retrieved from the document index, concatenated as context of the input prompt, and sent to the text generator. The RAG system allows language models to bypass retraining, enabling access to relevant information for reliable output generation and reduces fact conflict hallucination. A limitation of the LLM is that parametric knowledge in LLMs is static using a RAG system, a LLM can access knowledge that the model is not trained on, expanding its knowledgebase. (Lewis et al., 2021)



LLM Agents

Figure 9 - LLM agent overview (Wang et al., 2024)

As shown in figure 9, An agent uses external sources with large language models for various tasks like profiling, memory, planning, and actions. The profiling module allows LLMs to profile users in subcategories and show relevant information to the specific profiles. Planning modules allow LLMs to create a plan, spilt the plan into various subtasks and execute the plans. The memory module retains its previous behaviours and patterns, consolidates the memory for future uses. Agent is a powerful tool that let LLMs create better results without the need to retrain the model. The agent provides information to the context window of the language model. (Wang et al., 2024)



Agent Architectural design

LLMs can accomplish a wide range of tasks in form of Question answering. However, Agents need to evolve by leaning from the environment. To bridge this gap, design rational autonomous architecture assists LLMs to maximising their capabilities. The framework consists of profiling module, memory module, planning module and an action module. The profiling module identifies the role of an agent. Memory and planning enable the agent to recall past behaviours and plan future actions. The action module is responsible for executing the plans into specific outputs. Collectively the profiling module, memory module and planning module influence the action module. (Wang et al., 2024)



Profiling module

Agents perform tasks by assuming specific roles such as teachers, coders, doctors. The profiles are created by encompassing information such as age, gender, occupation, and psychological information. This information is written into the prompt influencing the LLMs behaviours. The agent profile information is determined by a specific application. There are three methods of creating and including agent profiles, Handicraft method, LLM generation method and Dataset alignment method. (Wang et al., 2024)

Handicraft method

Handicraft method uses the prompt submits additional information of various profiles, influencing the behaviour of the generated text. This creates agent personalities by using "you are an outgoing person" as a prompt. This method is flexible since one can assign any profile information to the agent. (Wang et al., 2024)

LLM generation method

During the LLM generation method, LLMs are leveraged to generate agent profiles. First creating manual agent profiles with backgrounds like age, gender and movie preferences and serve these profiles to the LLMs for few short trainings to generate more agent profiles based on the information. (Wang et al., 2024)

Dataset alignment method

Agent Profiles are obtained from real word datasets. Information about real humans can be organising in a dataset. The dataset can be connected to a natural language prompt leveraging the dataset to profile the agent. This process aligns the behaviours of the LLM with the real word humans. Encapsulating the attributes of real word population makes the agent behaviour more meaningful. (Wang et al., 2024)



Memory module

The memory module stores information perceived from the environment and uses the recoded memories to facilitate future actions. The memory module helps the agent accumulate experience, improves consistency and reliability The memory module is further divided into memory structure, memory format and memory operation.(Wang et al., 2024)



Memory structure

The LLMs incorporates principles of cognitive science research on human memory, The model registers input from short term memory and converts it into long term memory. The short-term memory for the agent retains information within the context window of the transformer. The long-term memory consolidates the information over an extended period and resembles the external vector storage from which the agent can retrieve information as needed. (Wang et al., 2024)

Unified memory

Simulates the short-term memory though in-context learning, the memory formulation from the in-context learning is directly written into the LLM prompts but due to the limitations of LLMs context window it is hard to pull all memories into a prompt which impairs the abilities of the agent system. (Wang et al., 2024)

Hybrid memory

Consists of a short term and a long-term memory. The short-term memory contains the context information about the agent's current behaviour and needs of the model. The information provided in the prompt can be considered as the short-term memory. The long-term memory consolidates important information like agents past behaviours and thoughts over time in a vector database which can be retrieved according to the current needs. The vector database encodes embeddings and stores the information of the past memory of the agent in the long-term memory. (Wang et al., 2024)



Memory format

There are multiple types of memory formats like natural language, embeddings, database, and structured lists. Each memory formats possess distinct strengths and weaknesses. Different memory formats are suitable for various applications. (Wang et al., 2024)





Natural language

In this format, agents' behaviour and observation are directly described using natural language. The memory information is expressed in a flexible manner and the rich semantic information can provide comprehensive signal to guide the behaviour of the agent's behaviour. Reflexion is one memory format that uses natural language. (Wang et al., 2024) Relflexion stores feedback in natural language and uses verbal reinforcement to help guide agent learn from prior failings. Reflexion converts binary feedback into verbal feedback in form of a textual summary, which is added to the context window of the LLM. Reflexion acts as a semantic signal providing agent with direction to improve on itself. Reflexion provides explicit hints for LLM actions and allows for targeted feedback. (Shinn et al., 2023)

Embeddings

In this format the information is encoded in an embedding matrix that enhances the memory retrieval. Memory segments or dialogue history can be embedded creating an indexed corpus for retrieval. (Wang et al., 2024)

Databases

The memory information is stored in a database, allowing the agent to alter the memories. The agent can utilize SQL statements to add, delete or revise information. The agents are fine tuned to understand and execute SQL queries, enabling them to interact with a database using natural language. (Wang et al., 2024)

Structured lists

Memory is organised into organised lists; the semantic meaning is conveyed in a concise manner using hierarchical tree structures. The tree structure encapsulates the relationship between the goals and the plans. (Wang et al., 2024)



Memory operation

Memory operation allows the agents to extract information from its memory to enhance the agents action. The memory interacts with the environment using the memory reading, writing and reflection. (Wang et al., 2024)

Memory reading

The objective of understanding the meaning of information stored in the memory to enhance the agent's actions. The key to good memory reading lies in how to extract valuable information for the history of actions. Relevancy, importance, and recency are three commonly used criteria for information extraction. (Wang et al., 2024)

Memory writing

Stores information about the perceived environment in the memory. Storing valuable information provides a foundation for retrieving valuable information when required. The two potential problems that should be carefully addressed during memory writing are, memory duplication that is storing information similar to existing memories and Memory overflow which removes information when the memory reaches its limit. (Wang et al., 2024)

Memory reflection

Agent capabilities to evaluate its cognitive, emotional, and behavioural processes. The objective of memory reflection is to provide agent with the capabilities to independently summarise abstract high-level information. The agent summarises its past memory into broader and abstract insights by asking key questions and obtain relevant information. Once the information is obtained, the agent generates abstract patterns and replaces all the original elements. (Wang et al., 2024)



Planning module

Simpler tasks and solving each task individually. This allows the agent to behave more responsibly and reliably. There is planning without feedback where the agent cannot influence decisions after an action is taken and there is planning with feedback that allows agents to create long horizontal plans to solve complex problems. (Wang et al., 2024)

Planning without feedback

Agent does not receive feedback and cannot influence the decisions made after an action is taken. The three most used planning techniques are single path reasoning, multi path reasoning and external planners.

Single path reasoning

As shown in figure 10, Single path reasoning splits final task into several smaller tasks. Single path reasoning tasks lead to only one subsequent step and follows smaller steps to achieve the final task. Chain of thought and HuggingGPT uses single path reasoning.



Figure 10 - Overview of single path reasoning. (Wang et al., 2024)

Chain of thought (COT)

COT is one of the most used single path reasoning agents. The first step is to create tasks that lead to only one subsequent step. The chain of thought then provides a window into the behaviour of the model, suggesting how the model arrived at a particular answer and provides opportunities to alter the reasoning path which went wrong. Chain of thought improves the commonsense reasoning abilities of a language model. Chain of though uses LLM Prompts to input few shots example, inspire LLMs to plan and act in steps. (Wei et al., 2022)

HuggingGPT



Figure 11 – Hugging GPT overview (Shen et al., 2023)

paper describes how ChatGPT can be used as a controller for HuggingFace external knowledge. Hugging face consists of multiple datasets trained in various AI tasks. The first step is to create a task plan, converting the user input into a task list and determining the execution order. Then selecting expert models on the HuggingFace platform based on the tasks. Retrieving information and executing tasks, finally generating a response. HuggingGPT uses ChatGPT as a decoder and the knowledgebase of HuggingFace as an encoder. (Shen et al., 2023)



Multi path reasoning



Figure 12 - Overview of multipath reasoning (Wang et al., 2024)

the reasoning step for generating the final plan are organised in a treelike structure. Each decision branches out and may have multiple subsequent steps. Individual steps may have multiple choices at each reasoning step. One of the most used multistep path reasoning is Self-consistent COT.

Self-consistent COT

Self-consistent COT believes that each problem has multiple ways of thinking, so it generates various reasoning paths and answers each branch using the CoT to decide on the most appropriate decision branch. Each nodes represent a thought which corresponds to a reasoning step. This approach resembles human thinking. (He et al., 2023)



External Planning

LLMs one shot training capabilities are effective but could be unreliable due to hallucination. To address this challenge research, turn to external planners. These tools are efficient search algorithms that identify the optimal plans. LLM+P is one of the external planners. (Wang et al., 2024)



Figure 12 – LLM+P overview (Liu et al., 2023)

As shown in figure 12, LLM+P creates task description using in context learning abilities of the LLMs. The long horizontal tasks create the plan in planning domain definition language (PDDL) format. The PDDL incorporates two files a domain file and a problem file. The domain file contains the state of the space and the actions with the preconditions and effect of these actions. The problem file states the initial stage and the goal conditions. The generated problem file and domain file is fed into the classic planner. The classic planner is used for creating a sequence of actions. The planning problems formalised are then solved by effective search algorhythms. The search algorhythms use planner as their search keywords to results answers. The results are then sent back to LLM. The LLM translates the results to a textual output. (Liu et al., 2023)



Planning with feedback

Agents create long horizontal plans to solve complex problems. Creating an effective plan directly from the beginning is difficult. Generating a cohesive plan requires various complex preconditions. Generally, just following the initial plan leads to failure. Making the initial plan non executable. Planning with feedback makes plans iterative and allows revising the plan based on feedback. (Wang et al., 2024)

Environmental feedback

Environmental feedback takes feedback form the objective world, for example the ReAct system makes observation from the agent's actions and voyager system uses Minecraft as its environment.



ReAct (Reason and act)



Figure 13 – ReAct overview (Yao et al., 2023)

ReAct aims to facilitate reasoning and planning for the agent though thought-act-observation.

Every text input from the context window is converted into a thought using LLMs. The thought describes what needs to be done to answer the text input. As shown in Figure 13, the react system acts on the thought using various tools like search engines. The output of the action is then observed by the LLM, if the observation requires more context, the observation is used to create the next sequence of thought-act-observation. Once the system is satisfied by the answer the system stops and provides an output. (Yao et al., 2023)

Voyager

As shown in figure 14, The Voyager system incorporates three types of environments feedback. Automatic curriculum, the Skill library, and the Iterative prompting mechanism. Automatic curriculum offers many benefits of open-ended exploration fostering a curiosity driven motivation for the agent to learn and explore from an internet scale knowledgebase. The voyager system naturally learns a variety of skills that it adds to its skill library.

Figure 14 - Voyager system overview (Wang et al., 2023)

The iterative prompting mechanism helps the skills in the skill library to improve automatically over time by environmental feedback, execution errors and self-verification of tasks. The three signals help agents make better plans for next actions. making the generated plans adaptive to the environment. (Wang et al., 2023)



Human feedback

Human feedback Involves taking feedback directly from humans that enhances the agents planning capabilities. This aligns the values and preferences of humans with the agent system. Taking feedback from humans and incorporating it in prompts. Different types of feedback can be combined to enhance the agent's planning capabilities. (Wang et al., 2024)



Model feedback

Model feedback Involves taking internal feedback from the agent themselves though pre-trained models. Self-refine, ChatCoT and Reflexion are some of the model feedback systems. Self-refine consists of three components, output, feedback, and refinement. An agent generates an output then it utilises LLMs to provide feedback to its own output. The output is then improved by feedback and refinement. Self-refine alters between feedback and refine step until a stopping condition is met. (Madaan et al., 2023)

ChatCoT

Figure 15 – ChatCoT Overview (Chen et al., 2023)

As shown in figure 15, ChatCoT Utilizes model feedback to improve the quality of its reasoning process through multiple rounds of conversations. The model feedback generates reasoning step that helps the model select or invoke a tool for execution. The LLM is provided with tool knowledge in the form of a description of tools and relevant tasks as a few shots example. Then the model is provided with reasons for selecting and executing a tool. (Chen et al., 2023)

Reflexion

Reflexion Enhances the planning though verbal feedback. The agent first produces an action based on its memory. The action is then saved in a verbal format. The textual summary is then added as context for the LLM for the next iteration. This self-reflection acts as a semantic signal that provides the agent with a direction to improve upon itself. (Shinn et al., 2023)



Action Module

The action model is influenced by the profile, memory and planning modules and directly interacts with its environment. The action module gathers information from multiple perspectives, Action goal, action production, action space and action impact. (Wang et al., 2024)

Action Goal

Agent performs various actions like task completion, communication, and environmental exploration.

5.4.1.1. Task completion this goal is aimed at completing a subset of specific tasks. Tasks have well defined objectives. Completion of each task contributes to the completion of the final task. (Wang et al., 2024)

5.4.1.2. Communication Actions communicate tasks with other agents or humans for collaboration or sharing information. (Wang et al., 2024)

5.4.1.3. Environmental exploration the agent explores unfamiliar environments to expand its knowledgebase. Voyager explores unknown skills and continuously refines the skills based on the environment feedback or trial and error. (Wang et al., 2023)



Action production

The agent takes actions through different strategies and sources this Differs from traditional LLMs where the models input, and output are associated with the actions for the agent. An agent can produce actions via memory recollection or plan following. (Wang et al., 2024)

5.4.2.1. Action via memory recollection the action is generated by extracting recent, relevant, and important information from agents’ memory according to the current task. If an action has been successfully completed, the agent retains that information and invokes the successful action for similar current tasks. The extracted memory is used as prompts to trigger actions. The agent may communicate with other agents to use another agent’s memory to complete a task. (Wang et al., 2024)

5.4.2.2. Action via plan following in this strategy the agent performs actions according to pre generated plans. The agent chooses a plan that has no plan failure signals. The agent solves each subgoals sequentially completing the final task. (Wang et al., 2024)



Action space

Actions that can be performed by an agent are referred to as action space. Action space can be of two types, external tools, and internal knowledge. (Wang et al., 2024)

5.4.3.1. External Tools to reduce hallucination the agents are empowered with the capabilities to call external tools for executing an action. There are multiple ways to implement these external tools like API, Database and knowledge base, External models. (Wang et al., 2024)

5.4.3.1.1.  APIs extend the action space and can be directly invoked by the LLM based on natural language or code input. HuggingGPT leverages intermodal corporation protocol to depend on expert models for complication of a tasks. Hugging GPT automatically creates plans and feeds task arguments into expert model to obtain interface results that it sends back to the LLM for execution. (Shen et al., 2023)

5.4.3.1.2. Database and knowledge base integrating external databases enables agents to obtain domain specific information that enables the model to generate more realistic actions. Using SQL statements to query databases facilitates realistic actions in a logical manner. (Wang et al., 2024)

5.4.3.1.3. External models can handle complicated tasks easily by corresponding to multiple APIs. TPTU is a model-based task planning that drafts a plan and provides helpful resources to the LLM for completion of a task. There are two types of TPTU, One step Agent (TPTU- OA) and Sequential Agent (TPTU-SA). TPTU-OA breaks down into subtasks in a single instant and performs all subtasks at once. TPTU-SA tackles current subtasks, upon successful resolution, the agent depends on the LLM to provide the next subtasks. With the help of various agents, the model can produce complicated text generation tasks like code generation, producing lyrics etc. (Ruan et al., 2023)



5.4.3.2. Internal knowledge many agents rely on internal knowledge of LLM to guide their actions. The LLM supports the agent by creating plans through which the agent behaves responsibly and effectively. LLMs internal knowledge enable planning capabilities, Conversational capabilities, and common-sense understanding capabilities. (Wang et al., 2024)

5.4.3.2.1. Planning capabilities LLMs have the capabilities of being a planner even without training. Offering one shot or a few short training courses can help build planning capabilities even further. LLMs are used as planners to decompress complicated tasks into simpler sub tasks. Voyager relies on the planning capabilities of the LLMs to complete tasks. (Wang et al., 2024)

5.4.3.2.2. Conversational capabilities LLMs can generate high quality conversations, that enable the agent to behave more like a human. The LLM can communicate feedback to the agent and the Agents can reflect on their own behavior using LLMs and improving their own capabilities. (Wang et al., 2024)

5.4.3.2.3. Common sense understanding capabilities LLMs can comprehend human like common senses. Based on this capability, agents can simulate human-like decisions. Generative agents can understand its current state and surrounding information and summaries high level observations. Without the common sense understanding of LLMs these behaviors cannot be simulated. (Wang et al., 2024)

Action Impact

The agent impact is the consequence of the actions of an agent.

5.4.4.1. Changing environments Agents can alter environments by performing actions like moving the position of an object or collecting information. Environments change from the actions of the agents. (Wang et al., 2024)

5.4.4.2. Altering internal states Actions taken by agents can also change the agent itself by updating its memory with additional information. This Enables agents to take actions and update understanding of an environment. (Wang et al., 2024)

5.4.4.3. Triggering new actions Agent actions can trigger another agent action, for example, TPU-SA triggers a subsequent plan only after the first plan of the agent is executed. The execution of the first plan gets updated to the model, triggering a new action. (Ruan et al., 2023)



5.4.5. Agent capabilities

Agent architecture allows LLMs to perform task, by adding various skills and experiences bridging the gap between LLMs and human preferences. There are various strategies though which agents acquire its capabilities, though fine tuning and capabilities acquired without fine tuning. (Wang et al., 2024)

5.4.5.1. Capabilities with fine tuning Fine tuning enhances agent's capabilities to complete a task efficiently. Fine tuning datasets can be based on human annotation, LLM generated or with real world datasets.



5.4.5.1.1. Fine tuning with human annotated dataset Researchers, design an annotated task and then recruit workers to complete tasks. The agent understands human preferences though the feedback provided by the recruits. Chain of Hindsight (CoH) uses supervised fine training and Reinforcement learning from human feedback to enhance the performance of large language models, aligning the LLMs with human values and preferences by giving rich detailed feedback in form of natural language. The Agent converts natural language into structured memory efficiently. (Ouyang et al., 2022)

5.4.5.1.2. Fine tuning with LLM generated dataset

Figure 16 – ToolLLaMA overview (Deng et al., 2023)

As shown in figure 16, ToolLLaMA is a fine tuning with LLM generated dataset model, the agents use LLMs for annotation. Using LLM generated datasets can annotate many samples and is more cost effective when compared to human annotation. The drawback of LLM generated datasets are that they are not as reliable as the human feedback fine tuning models. Toolbench consists of API collection, instruction generation and solution path annotation.

API collection facilitates 1600+ APIs spanning in 49 categories to integrating various tools to accomplish complex tasks.

The instruction generation is based on the API retriever and the toolEval. API retriever recommends relevant APIs using BERT by encoding the instructions and the API documents into two embeddings and calculate their relevancy with embedding similarity. ToolEval uses Pass rate that marks the successful completion of a task under a predefined criteria and win rate that compare two solution path and evaluates and obtains the preference.

The solution path annotation uses Depth first search-based decision tree (DFSDT) to bolster the planning and reasoning abilities of LLMs. DFSDT uses multipath reasoning and significantly improves the annotation efficiency of LLMs. Solution path annotation contains multiples rounds of model reasoning and real time API calls to derive the final response. ToolLLaMA is Finetuning LLaMA Based on toolbench. ToolLLama demonstrates ability to execute complex instruction and generation to unseen APIs demonstrating zero shot generalisation abelites. (Qin et al., 2023)

5.4.5.1.3. Fine tuning with real world dataset Directly using real world dataset to fine tune, the agent is also a common strategy. Mind2Web builds a genderized agent that follows language instructions and corresponding tasks to enhance the agent's capabilities in the web domain. Mind2Web can plug into an existing LLM and directly acquire information from a website and carry out actions on HTML based websites. Mind2Web is trained on 137 websites that spans 31 different domains and can perform over 2000 tasks. Mind2Web allows the agents to preform various actions like booking a flight ticket, extract real time weather information. (Deng et al., 2023)



5.4.5.2. Capabilities without fine tuning There are two methods to unleash agent capabilities prompt engendering and mechanism engineering. Prompt engineering is the process of writing valuable information into prompts to enhance the model capabilities or unleash existing LLM capabilities. Mechanism engineering involves developing specialised modules introducing strategies to enhance agent capabilities.

5.4.5.2.1. Prompt engineering Due to strong language comprehension capabilities of LLMs, People can interact with LLMs using natural language. Users can describe desired capabilities to influence LLM actions. COT empowers agents with the capabilities for complicated task reasoning. By presenting reasoning steps as a few short examples using the prompts. (Wei et al., 2022)



5.4.5.2.2. Mechanism engineering There are multiple methods mechanism engineering can be performed there is trial and error method, Crowd sourcing, Experience accumulation and Self driven evolution.

5.4.5.2.2.1. Trial and error in this method, the agent performs an action and a predefined critic, judges each action. If the action is unsatisfactory the agent reacts by incorporating the critic's feedback. In RAH, the agent first generates a response then compares it with human feedback. If the predicted response and the human feedback differs then the critic generate a failure response. The failure or success helps fine tune the model. Generating feedback closer to human feedback. (Shu et al., 2024)





5.4.5.2.2.2. Crowd sourcing



Figure 17 – Crowd sourcing overview (Du et al., 2023)

As shown in figure 17, Crowd sourcing Organises debating mechanism that leverages the wisdom of crowds to enhance agent capabilities.

If their responses are not consistent, they will be prompted to incorporate the solutions from other agents and provide an updated response. This iterative process continues until reaching a final consensus answer. Multi-agent debate is an automatic evaluation method which compares different instances of the same language model, and the final answer generated is more factually accurate and solves reasoning questions more accurately. Round of Debate uses multiple rounds of debate for multiple agents/ instances of the same language model. Compares their knowledge by sharing their response to the other instance/agent, updating the incorrect response in multiple rounds of questions and answers. Agent number is the number of agents or instance of the same language model used to compare knowledge. The number of agents and the number of rounds can be controlled according to the use case. The Study then goes further in explaining how multiple agents' multiple instances of different large language models could be. Comparing knowledge improves the performance of both agents. An LLM agent can offload tasks on behalf of the user. (Du et al., 2023)

5.4.5.2.2.3. Experience accumulation Agents are not given any instructions to do to solve tasks in the beginning, they explore the environment until they successfully accomplish a task. Once a task is successfully accomplished the agent store the methods of completing that task in the memory. If the agent encounters a similar task, then the relevant memories are extracted to complete that task. The designed memory accumulation allows agents to utilities its memory according to the environment feedback and agent self-verification results. (Wang et al., 2023)

5.4.5.2.2.4. Self-driven evolution in this method the agent automatically set goals and gradually improve capabilities by exploring the environment and receive feedback from the reward function. Following this mechanism the agent acquires knowledge and develops capabilities according to its own preferences. (Wang et al., 2024)



Research Triangulation

Triangulation is a method used to increase the credibility and validity of research findings though multiple methods in this paper we are focusing on data triangulation, investigator triangulation, methods triangulation. Using multiple triangulation methods are together is known as methodological triangulation. (Mathison, 1988)

6.1. Data triangulation refers to the use of multiple data sources to enhance the validity of the research findings. Data triangulation may also require sources from various times and space. For example, to observe the effect of a program, one should observe the students at different time of schools and in different settings. (Mathison, 1988)

6.2.  Investor Triangulation involves more than one investigator to accomplish data collection to reduces individual biases of the investigator. (Mathison, 1988)

6.3. Method Triangulation involves the use of various methods of research like interview, observation, field notes, surveys. (Mathison, 1988)

6.4. Methodological triangulation refers to the use of multiple methods to examine a phenomenon. The assumption is that using multiple data sources, Investigators and methods should result in a single claim because biases naturally cancelled out. This assumption suggests that when triangulation strategy is used, the results will be a converge into a single perspective of a phenomena. However, there might be inconsistency, or the results may be contradictory. Convergence when data from various sources or methods agree the outcomes is convergence. Inconsistency when the data from various sources or methods differ based on the perspective. In other words, the data obtained may not be confirming nor contradicting. Contradictory is when the researcher has incommensurable propositions, in other words when the data from different methods does not converge.  (Mathison, 1988)



Research question

How to reduce AI hallucination using the principles of research triangulation and LLM agents?





Research findings

There are multiple profiling, memory, planning and action strategies and each strategy is suitable for a tasks. It becomes important to understand the most appropriate strategy for a given task to ensure efficiency and reliability.

For example, action via Memory recollection and action via plan following are two different action production methods. Given an identical task, the output for both methods would produce different results. For some tasks, action via memory recollection will produce better results and for others, action via plan following will produce better results. Only after comparing the results of each method would we be able to understand the appropriate method of a task. The Information model aims to determining the best strategy for each task. The information model compares multiple methods and compares knowledge from multiple sources to draw conclusions, using the concept of methodological triangulation aiming to reduce fact conflict hallucination by focusing on hallucination from data and training.

Figure 18 – Information model overview



Information model

With hundreds of agents, it can be hard to determine which agent is best for a specific task, each agent specializes in a task and contains a specific kind of knowledge. Agents use various methods to complete a task. For example, in the planning module there is planning without feedback which contains single path reasoning and multi path reasoning and planning with feedback which contains environmental feedback, human feedback and model feedback. There are multiple ways of completing a task but only one method is the most suitable. Some tasks may benefit from planning with feedback and others with planning without feedback. The information model aims to automate agent selection using machine learning. Information model as an external model would guide the large language model to select the most appropriate agent. The information model can be used in LLMs and can also be used in other transformer based generative AI tasks like image generation, music and audio generation.

The way a large language model selects the best token, an agent selection system would select the best agent for a task.

It is able to select the best agent using weights. The weights determine the selection of the agent for a specific task. The agents with higher weights are more likely to be selected for a task. As shown in figure 18, The Agents are weighted in three matrices, Task matrix, Category matrix and Library matrix.

The Large language model would continuously update the weights of the library matrix during interface which would gradually make the model select the best agent for each request.

Weights



Figure 19 – Overview of weight updation

Agent Comparison

As shown in figure 19, Agent would be assigned weights for tasks, categories and library matrices. Weights would be determined according to the performance of the agent. The weights of agents with the desired output would increase while the weights of agents with incorrect output would reduce. With extensive training on multiple tasks the best agents for each category would be determined.

Desired output

The desired output is when multiple agents come to the same conclusion or are consistent. If the agents are inconstant or contradictory, human feedback would be required to determine the correct answer. The process of comparing multiple agents in the task matrix simulate investigator triangulation, the process of comparing multiple methods in the category matrix simulate method triangulation. Comparing data of multiple agents in the library matrix simulate data triangulation. Here multiple methods are used to examine a phenomenon to converge into a single perspective which simulate methodological triangulation.

Comparing multiple perspectives and coming to a single clam cancels out biases. The use of methodological triangulation may reduce fact conflict hallucination even further.

Model feedback During Convergence

The agents with the highest weights would use crowd sourcing and multi agent debate to conclude to single perspective. Model feedback through crowd sourcing compares knowledge of agents against other agents. For example, Agent 1, Agent 2, Agent 3, and Agent 4 are made to give an output. Agents 1, 2 and 3 outputs are similar, but Agent 4 results do not match. Assuming agent 1 and 3 are the once with the highest weights, they will go through multiple rounds of debate until they draw to the same conclusion. The weights of Agent 4 in this case will be reduced while the weights of the other 3 will increase.

Human feedback during Inconsistent or contradictory

A human determines the desired output if the model marks the result Inconsistent or contradictory the weights of the agents that provide the desired output according to the human feedback increases while the weights of the agents that do not provide the desired output decrease.

Assigning weights

Weights are assigned to the agent using human feedback or model feedback in each category. The agents that are not capable of producing are marked and their weights are unchanged. The weights of the agents that produce a positive feedback increase and the weights of the agents that produce negative feedback reduce.

Adding weights

The assigned weights are then added to the original weights of each agent, updating the reliability of the agent for each task.

Task matrix

Agents contain specialized knowledge suited for specific tasks. The task matrix holds knowledge of the agent’s specialization in each task. The task matrix contains weights of agents for various tasks and determines the most suited method. The weights of the task matrix are determined by the category and library matrix.

Category matrix

The category matrix marks the weights of the agents in 5 categories, 4 static categories planning, memory, action, and profiling and one dynamic category library matrix. The total score of the category matrix is added to the task matrix. Using the agent description, the model checks if an agent can complete the task. An agent may or may not have the capabilities to provide results in all the categories listed in the category matrix. For example, ReAct does not have profiling or memory capabilities while voyager does not have profiling capabilities. It is important to make sure the information model does not provide weights to a category the agent does not operate in. for example the model should not provide weights to Voyagers profiling capabilities. Once the agent categories are selected, the capable agents are asked to provide an output. The outputs are then compared with other agents and depending on the accuracy of the output, weights are provided to each agent. If the agent can output the desired result, the agents’ weights increase for that category.



Static categories

The static categories planning, memory, action, and profiling would not be updated during interface as they are methods to complete a task. During training once the best method is determined, the model would check and use the weights of the highest weighted agents. For example, a task that benefits from planning without feedback would always show best results using the same method.

Dynamic category

The library matrix is a dynamic category as it updates its weights during training and during interface. It is important for the library category to update its weights during interface as the information of the agents may update with time which may alter the selection of the agent. As shown in figure 20, The library category is further divided into subcategories in the library matrix. The subcategories of the library matrix are specified and no new category is added once the categories are specified. The library matrix determines the agent's quality of knowledge. For example, for a task that requires healthcare information, the model will check the weights of the healthcare source and select the agent with the highest weights. The agent with the highest weights would most likely to be selected. The library matrix would compare knowledge of agents against other agents and use crowdsourcing and multiagent debate to come to a single conclusion. A single source with the highest weights may or may not be most suited for the task due to the limitations of the knowledge of one agent. hence it is important for the library category to use multi agent debate to draw a single conclusion from multiple sources. The sum of all libraries is added to the library category in the category matrix providing the category matrix the position of the most desired agents in the matrix.

The library matrix can be used for agents as well as compare different knowledge from multiple RAG systems further expanding the knowledge of the information model.



Figure 20 – library matrix

Sum of category matrix

The sum of planning, memory, action, profiling and library are added to the task matrix, The total weights determine the best suited agent for that task. Providing the position of the desired agents for a task in the task matrix.

During training

During training the agents compare outputs of multiple agents and once the system is ready with the weights a frequent patten analysis is done on the information model to reduce the number of tasks in the task matrix, making it easier for the large language model to go through the tasks during interface.

Comparing agent outputs

multiple agents are given a task to complete. The agent description is used to check if the agent can complete that task. The list of agents that can complete that task are then asked to execute an output. The output of all the agents that can execute the task are then compared with each other. If multiple agents are able to come to a single conclusion the model marks the task as converging. The weights of agents able to converge are positively weighted, while the contradictory agents are negatively weighted. The best agent for that task would likely have the highest weights with multiple rounds of training. If the model is unable to come to convergence human feedback determines the weights for that round. This process is repeated for multiple tasks.

Frequent pattern analysis on Information Model

During training, the task list in the task matrix would increase. The task may be too large and contain repetitive tasks. The repetitive tasks can be group together by finding similarities. For example, A task that requires the agent to extract information from Amazon and a task that requires the model to extract information from Walmart are similar and would most likely use the same planning methodology. Here similar tasks can be grouped together to make the task matrix smaller in size and simplifying procedure selection. Performing Frequent pattern analysis on the Task matrix would make the matrix concise, shortening its size while retaining crucial information.

During interface

When a large language model is assigned a task, The assigned task would be compared with similar tasks in the task matrix. The weights of the task most similar to the assigned task would be selected. The comparison can be done with the transformer architecture, by using the query and key matrix as the assigned task and the value matrix as the information model to find similarities between the assigned tasks and the information model tasks. After a similar task is found, the agents with the highest weights would be used during interface. If the task requires the dynamic category, the agents with the highest weights in the library category would be compared using multi agent debate during interface to find the best agent and the weights of the agents would be updated according to the agent’s knowledge during interface, gradually making the library category better with each request.



Sources

Chen, X., Song, D., Gui, H., Wang, C., Zhang, N., Jiang, Y., Huang, F., Lv, C., Zhang, D., & Chen, H. (2024). FactCHD: Benchmarking fact-conflicting hallucination detection.



Chen, Z., Zhou, K., Zhang, B., Gong, Z., Zhao, W., & Wen, J.-R. (2023). ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models.



Deng, X., Gu, Y., Zheng, B., Chen, S., Stevens, S., Wang, B., Sun, H., & Su, Y. (2023). MIND2WEB: Towards a Generalist Agent for the Web.



Du, Y., Li, S., Torralba, A., Tenenbaum, J., Mordatch, I., & Brain, G. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate.



He, L., Li, Z., Cai, X., & Wang, P. (2023). Multi-Modal Latent Space Learning for Chain-of-Thought Reasoning in Language Models. Proceedings of the AAAI Conference on Artificial Intelligence, 38(16), 18180–18187.



Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Madotto, A., & Fung, P. (2022). Survey of Hallucination in Natural Language Generation. ACM Computing Surveys, 55(12).



Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-T., Rocktäschel, T., Riedel, S., Kiela, D., Facebook, & Research, A. (2021). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.



Liu, B., Jiang, Y., Zhang, X., Liu, Q., Zhang, S., Biswas, J., & Stone, P. (2023). LLM+P: Empowering Large Language Models with Optimal Planning Proficiency.



Mathison, S. (1988). Why Triangulate? Educational Researcher, 17(2), 13–17.



Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., Gupta, S., Majumder, B., Hermann, K., Welleck, S., Yazdanbakhsh, A., & Clark, P. (2023). SELF-REFINE: Iterative Refinement with Self-Feedback.



Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W.-T., Koh, P., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). FACTSCORE: Fine-grained atomic evaluation of factual precision in long form text generation.



Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback.



Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., Zhao, S., Hong, L., Tian, R., Xie, R., Zhou, J., Gerstein, M., Li, D., Liu, Z., & Sun, M. (2023). TOOLLLM: FACILITATING LARGE LANGUAGE MODELS TO MASTER 16000+ REAL-WORLD APIS.



Ruan, J., Chen, Y., Zhang, B., Xu, Z., Bao, T., Du, G., Shi, S., Mao, H., Li, Z., Zeng, X., Zhao, R., & Research, S. (2023). TPTU: Large Language Model-based AI Agents for Task Planning and Tool Usage.



Shen, Y., Song, K., Tan, X., Li, D., Lu, W., Zhuang, Y., University, Z., & Research, M. (2023). HuggingGPT: Solving AI tasks with chatgpt and its friends in hugging face.



Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning.



Shu, Y., Zhang, H., Gu, H., Zhang, P., Lu, T., Li, D., & Gu, N. (2024). RAH! RecSys–Assistant–Human: A Human-Centered Recommendation Framework With LLM Agents. IEEE Transactions on Computational Social Systems, 1–12.



Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need.



Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, J., Anandkumar, A., Nvidia, & Caltech. (2023). VOYAGER: An Open-Ended Embodied Agent with Large Language Models.



Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., Wayne Xin Zhao, Wei, Z., & Wen, J. (2024). A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6).



Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi Quoc, E., Le, V., & Zhou, D. (2022). Chain-of-Thought prompting elicits reasoning in large language models chain-of-thought prompting.



Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS.



Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang, X., Zhao, E., Zhang, Y., Chen, Y., Longyue, Luu, A., Bi, W., Shi, F., Shi, S., & Lab, A. (2023). Siren’s Song in the AI Ocean: A Survey on Hallucination in Large Language Models.
