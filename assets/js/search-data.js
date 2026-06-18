// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-projects",
          title: "projects",
          description: "From-scratch implementations tracing the evolution of neural networks.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "post-speculative-decoding-trading-draft-guesses-for-parallel-verification",
        
          title: "Speculative Decoding: Trading Draft Guesses for Parallel Verification",
        
        description: "How speculative decoding uses a small draft model to propose tokens and a large model to verify them in one pass, achieving 2-3x speedup on GPU.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/speculative-decoding/";
          
        },
      },{id: "post-prefix-caching-reusing-kv-across-requests",
        
          title: "Prefix Caching: Reusing KV Across Requests",
        
        description: "Why every request recomputes the same system prompt, and how prefix caching eliminates redundant prefill by sharing KV pages.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/prefix-caching/";
          
        },
      },{id: "post-flash-attention-writing-a-gpu-kernel-in-triton",
        
          title: "Flash Attention: Writing a GPU Kernel in Triton",
        
        description: "How Flash Attention avoids materializing the O(T²) attention matrix by tiling Q×K^T in GPU SRAM, with a from-scratch Triton kernel.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/flash-attention-from-scratch/";
          
        },
      },{id: "post-pagedattention-virtual-memory-for-kv-cache",
        
          title: "PagedAttention: Virtual Memory for KV-Cache",
        
        description: "Why torch.cat creates O(n²) memory copies, and how PagedAttention eliminates them with pre-allocated page pools.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/pagedattention-from-scratch/";
          
        },
      },{id: "post-kv-cache-why-naive-inference-is-o-n-and-how-caching-fixes-it",
        
          title: "KV-Cache: Why Naive Inference is O(n²) and How Caching Fixes It",
        
        description: "How KV-cache reduces autoregressive generation from O(n²) to O(n), with from-scratch implementation and 2.6x speedup benchmarks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/kv-cache-from-scratch/";
          
        },
      },{id: "post-gpt-2-loading-real-weights-and-generating-text",
        
          title: "GPT-2: Loading Real Weights and Generating Text",
        
        description: "Writing GPT-2&#39;s forward pass from scratch, loading OpenAI&#39;s 124M weights, and generating coherent English text.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/gpt2-from-scratch/";
          
        },
      },{id: "post-transformer-self-attention-from-scratch",
        
          title: "Transformer: Self-Attention From Scratch",
        
        description: "Building a transformer from scratch — Q/K/V attention, multi-head, positional encoding, and why it replaced RNNs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/transformer-from-scratch/";
          
        },
      },{id: "post-rnn-and-lstm-sequence-memory-from-scratch",
        
          title: "RNN and LSTM: Sequence Memory From Scratch",
        
        description: "Building RNN and LSTM from scratch — hidden states, vanishing gradients, and why LSTM gates fix them.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/rnn-lstm-from-scratch/";
          
        },
      },{id: "post-word2vec-how-words-become-vectors",
        
          title: "Word2Vec: How Words Become Vectors",
        
        description: "Building skip-gram Word2Vec from scratch — embedding lookup tables, backpropagation, and cosine similarity.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/word2vec-from-scratch/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "projects-transformer-from-scratch",
          title: 'Transformer From Scratch',
          description: "Full transformer encoder-decoder — self-attention, multi-head attention, positional encoding, layer normalization.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/attention_from_scratch/";
            },},{id: "projects-gpt-2-from-scratch",
          title: 'GPT-2 From Scratch',
          description: "Complete GPT-2 (124M) forward pass from scratch — loading real OpenAI weights, generating coherent text.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/gpt2_from_scratch/";
            },},{id: "projects-inference-engine",
          title: 'Inference Engine',
          description: "From-scratch inference engine — KV-cache (2.6x), PagedAttention, Flash Attention (Triton GPU kernel), prefix caching, continuous batching.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/inference_engine/";
            },},{id: "projects-rnn-amp-lstm-from-scratch",
          title: 'RNN &amp;amp; LSTM From Scratch',
          description: "RNN and LSTM with Bahdanau Attention — character-level text generation, vanishing gradient analysis.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/rnn_from_scratch/";
            },},{id: "projects-word2vec-from-scratch",
          title: 'Word2Vec From Scratch',
          description: "Word2Vec skip-gram from scratch — embeddings, backpropagation, cosine similarity, custom training.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/word2vec/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-science-fundamentals/";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/introduction-to-machine-learning/";
            },},{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/aserputov", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/serputov", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
