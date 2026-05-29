export const SAMPLE_PARSED_PAPER = {
  arxivId: "2005.11401",
  title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
  authors: [
    "Patrick Lewis",
    "Ethan Perez",
    "Aleksandra Piktus",
    "Fabio Petroni",
    "Vladimir Karpukhin"
  ],
  year: 2020,
  abstract:
    "Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation. We endow the models with access to a dense vector index of Wikipedia, and train them in an end-to-end fashion. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, and another which can use different passages per token. We set the state-of-the-art on three open domain QA tasks.",
  sections: [
    {
      title: "Introduction",
      content:
        "Language models trained on large corpora have demonstrated strong performance across many NLP tasks. These models store knowledge implicitly in their parameters. However, this approach has several limitations: knowledge becomes stale after training, the model cannot cite sources, and updating knowledge requires expensive retraining. We propose RAG, which combines a parametric memory (a pre-trained seq2seq model) with a non-parametric memory (a dense vector index of Wikipedia). The retriever provides relevant documents given a query, and the generator produces output conditioned on the query and the retrieved documents. Our approach enables models to generate more specific, diverse, and factual language. We demonstrate this across open-domain question answering, abstractive question answering, Jeopardy question generation, and fact verification."
    },
    {
      title: "Methods",
      content:
        "RAG models use a pre-trained neural retriever to retrieve documents, then condition a pre-trained seq2seq model (the generator) on these retrieved documents. The retriever p_eta(z|x) returns top-K truncated distributions over text passages given a query x. The document index uses dense inner product search (MIPS). We explore two RAG formulations: RAG-Sequence uses the same retrieved document to generate the complete sequence — the model retrieves K documents and generates the output sequence for each document, then marginalizes over the documents. RAG-Token can use different documents for each generated token, allowing the generator to retrieve different evidence for different parts of the answer. Both use DPR (Dense Passage Retrieval) as the retriever and BART-large as the generator. Documents are encoded using a bi-encoder architecture."
    },
    {
      title: "Results",
      content:
        "On open-domain QA benchmarks: Natural Questions 44.5 EM (new state of the art), TriviaQA 56.8 EM (competitive with T5), WebQuestions 45.5 EM (new state of the art). RAG models generate more factual and specific text compared to purely parametric models. On fact verification (FEVER), RAG achieves 70.8% label accuracy. Analysis shows that larger document indices consistently improve performance, suggesting the model effectively uses the retrieved information rather than ignoring it."
    },
    {
      title: "Discussion",
      content:
        "RAG represents a promising direction for knowledge-intensive tasks where knowledge needs to be updatable without retraining, provenance and citations are important, and the task requires combining information from multiple sources. Limitations include the computational cost of retrieval, the quality of the document index, and the challenge of handling conflicting information from different retrieved documents. Future work includes scaling the document index, improving the retriever, and exploring end-to-end training of the full system."
    }
  ],
  references: [
    "Karpukhin et al. (2020) - Dense Passage Retrieval for Open-Domain QA",
    "Lewis et al. (2019) - BART: Denoising Sequence-to-Sequence Pre-training",
    "Guu et al. (2020) - REALM: Retrieval-Augmented Language Model Pre-Training"
  ]
};

export type ParsedPaperSample = typeof SAMPLE_PARSED_PAPER;
