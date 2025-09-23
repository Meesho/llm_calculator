# Sample Dataset Format

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [optimal_llm_calculator/sample-dataset.csv](optimal_llm_calculator/sample-dataset.csv)

</details>



## Purpose and Scope

This document describes the structure and format of the sample dataset used by the LLM Runtime Calculator to estimate optimal training parameters. The sample dataset serves as a representative example for calculating tokenization metrics, data processing times, and memory requirements during fine-tuning operations.

For information about the runtime calculator implementation, see [Runtime Calculator Script](#2.1). For details about the training dataset used in actual model fine-tuning, see [Training Dataset](#3.5).

## Dataset Structure Overview

The sample dataset is provided as a CSV file located at [optimal_llm_calculator/sample-dataset.csv:1-134]() and follows a standardized two-column format designed for sentiment classification tasks.

### CSV Schema

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `text` | String | Input prompt containing the instruction and tweet text |
| `output` | String | Target sentiment classification label |

### Data Format Specification

```mermaid
flowchart TD
    CSV[sample-dataset.csv] --> HEADER["Column Headers: text, output"]
    HEADER --> ROWS["133 Data Rows"]
    
    ROWS --> TEXT_COL["text Column"]
    ROWS --> OUTPUT_COL["output Column"] 
    
    TEXT_COL --> PROMPT_FORMAT["Detect the sentiment of the tweet. [TWEET_CONTENT]"]
    OUTPUT_COL --> LABELS["Positive | Negative"]
    
    PROMPT_FORMAT --> INSTRUCTION["Instruction Text"]
    PROMPT_FORMAT --> TWEET_CONTENT["Cryptocurrency/Bitcoin Tweet"]
    
    LABELS --> POSITIVE["Positive Sentiment"]
    LABELS --> NEGATIVE["Negative Sentiment"]
```

**Sources:** [optimal_llm_calculator/sample-dataset.csv:1-134]()

## Column Specifications

### Text Column Format

Each entry in the `text` column follows a consistent prompt template:

```
"Detect the sentiment of the tweet. [TWEET_CONTENT]"
```

The prompt structure consists of:
- **Instruction prefix**: `"Detect the sentiment of the tweet. "`
- **Tweet content**: Variable-length cryptocurrency-related social media text
- **Content characteristics**: Contains mentions of Bitcoin, blockchain, trading, and related financial topics

Example entries from the dataset:
- `"Detect the sentiment of the tweet. RT @tippereconomy: Another use case for #blockchain and #Tipper..."`
- `"Detect the sentiment of the tweet. Bitcoin heading back down. … $BTCUSD"`

### Output Column Format

The `output` column contains binary sentiment classifications:
- **Positive**: Indicates favorable or optimistic sentiment
- **Negative**: Indicates unfavorable or pessimistic sentiment

## Dataset Processing Workflow

```mermaid
flowchart LR
    SAMPLE_CSV[sample-dataset.csv] --> TOKENIZER["LlamaTokenizerFast"]
    TOKENIZER --> TOKEN_COUNT["Token Count Analysis"]
    TOKEN_COUNT --> CALC_ENGINE["optimal_runtime_calculator.py"]
    
    CALC_ENGINE --> BATCH_SIZE["datapoints_in_1_step Calculation"]
    CALC_ENGINE --> TIME_EST["Training Time Estimation"]
    
    BATCH_SIZE --> OPTIMAL_PARAMS["Optimal Runtime Parameters"]
    TIME_EST --> OPTIMAL_PARAMS
    
    subgraph "Text Processing"
        TOKENIZER
        TOKEN_COUNT
    end
    
    subgraph "Runtime Calculation"
        CALC_ENGINE
        BATCH_SIZE
        TIME_EST
    end
```

**Sources:** [optimal_llm_calculator/sample-dataset.csv:1-134]()

## Data Characteristics

### Content Distribution

The dataset exhibits the following characteristics:

| Metric | Value | Description |
|--------|--------|-------------|
| Total Rows | 133 | Complete dataset size |
| Text Format | Instruction + Tweet | Consistent prompt template |
| Domain | Cryptocurrency/Bitcoin | Specialized content domain |
| Task Type | Binary Classification | Sentiment analysis task |

### Prompt Structure Analysis

```mermaid
graph TB
    DATASET[sample-dataset.csv] --> ROW_ANALYSIS["Row-by-Row Analysis"]
    ROW_ANALYSIS --> INSTRUCTION_PART["Instruction: 'Detect the sentiment of the tweet.'"]
    ROW_ANALYSIS --> TWEET_PART["Tweet Content: Variable Length"]
    
    INSTRUCTION_PART --> FIXED_TOKENS["Fixed Token Count per Row"]
    TWEET_PART --> VARIABLE_TOKENS["Variable Token Count per Row"]
    
    FIXED_TOKENS --> TOKEN_CALC["Total Token Calculation"]
    VARIABLE_TOKENS --> TOKEN_CALC
    
    TOKEN_CALC --> RUNTIME_EST["Runtime Estimation Input"]
    
    subgraph "Tokenization Impact"
        FIXED_TOKENS
        VARIABLE_TOKENS
        TOKEN_CALC
    end
```

**Sources:** [optimal_llm_calculator/sample-dataset.csv:1-134]()

### Label Distribution

Based on the dataset content:
- **Positive samples**: Majority of entries (approximately 85%)
- **Negative samples**: Minority of entries (approximately 15%)
- **Class imbalance**: Present, reflecting real-world social media sentiment patterns

## Usage in Runtime Calculations

The sample dataset serves as input to the runtime calculator through the following process:

1. **Tokenization**: Each `text` entry is processed by `LlamaTokenizerFast` to determine token counts
2. **Batch calculation**: Token counts inform the `datapoints_in_1_step` calculation
3. **Time estimation**: Dataset size and complexity influence training time projections
4. **Memory estimation**: Text length variations help estimate memory requirements

The dataset's cryptocurrency focus and instruction-following format make it representative of typical fine-tuning scenarios for sentiment analysis tasks on social media content.

**Sources:** [optimal_llm_calculator/sample-dataset.csv:1-134]()
