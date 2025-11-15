# Part 2: Text-to-SQL Implementation Guide

## 📋 Summary of What We Implemented

I've implemented the **baseline approach** for fine-tuning T5 on text-to-SQL task. This should achieve ≥65 F1 on the test set.

### Files Modified:
1. ✅ **load_data.py** - Data loading and preprocessing
2. ✅ **t5_utils.py** - Model initialization, saving, and loading
3. ✅ **train_t5.py** - Evaluation and test inference functions

---

## 🎯 WHAT We're Doing (High-Level)

```
Natural Language Query → T5 Model → SQL Query → Execute on Database → Records
```

**Example:**
- **Input**: "What flights are there from Boston to Baltimore?"
- **Output**: `SELECT * FROM flight WHERE from_airport = 'BOS' AND to_airport = 'BWI'`
- **Execution**: Returns actual flight records from the database

---

## 📝 Detailed Implementation Breakdown

### 1️⃣ **Data Processing (load_data.py)**

#### **WHAT**: Convert text files to tokenized tensors

#### **WHY**:
- Neural networks work with numbers, not text
- T5 needs special format: input IDs, attention masks, decoder inputs/targets
- Different lengths need padding for batch processing

#### **HOW**:

```python
# Step 1: Load files
train.nl → ["show me flights...", "what is..."]
train.sql → ["SELECT * FROM...", "SELECT fare..."]

# Step 2: Add task prefix (T5 was pretrained with this!)
"show me flights" → "translate English to SQL: show me flights"

# Step 3: Tokenize with T5 tokenizer
"translate English to SQL: show me flights" → [3337, 1566, 12, 4521, 10, ...]

# Step 4: Create decoder inputs (teacher forcing)
SQL: [50, 100, 200, 300, 1]  (1 = EOS)
Decoder input:  [0, 50, 100, 200, 300]  (0 = start token)
Decoder target: [50, 100, 200, 300, 1]  (what we predict)
```

#### **Key Functions**:

- **`T5Dataset.__init__()`**:
  - Loads T5 tokenizer
  - Calls `process_data()`

- **`process_data()`**:
  - Reads `.nl` and `.sql` files
  - Tokenizes with task prefix
  - Creates shifted decoder inputs for teacher forcing

- **`normal_collate_fn()`**:
  - Pads sequences to same length (dynamic padding!)
  - Creates attention masks
  - Returns batched tensors

- **`test_collate_fn()`**:
  - Same as normal but no targets (test has no labels)

#### **ALTERNATIVES**:
| What We Did | Alternative | Trade-off |
|-------------|-------------|-----------|
| Add task prefix | No prefix | Prefix helps model understand task |
| Dynamic padding | Fixed padding | Dynamic saves compute |
| Preprocess all data | On-the-fly | Preprocess is faster during training |

---

### 2️⃣ **Model Initialization (t5_utils.py)**

#### **WHAT**: Load pretrained T5-small model

#### **WHY**:
- Pretrained weights = already understands language
- T5-small = 60M parameters (good balance)
- Starting from scratch would need much more data/time

#### **HOW**:

```python
# For fine-tuning (BASELINE):
model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
# This loads weights pretrained on C4 dataset (massive text corpus)

# For training from scratch (Extra Credit):
config = T5Config.from_pretrained('google-t5/t5-small')
model = T5ForConditionalGeneration(config)
# This creates same architecture but with random weights
```

#### **Key Functions**:

- **`initialize_model()`**:
  - Loads pretrained T5-small if `--finetune` flag is set
  - Otherwise initializes from scratch
  - Moves model to GPU

- **`save_model()`**:
  - Saves model weights (state_dict) to disk
  - Saves both "best" (highest F1) and "last" (most recent)

- **`load_model_from_checkpoint()`**:
  - Recreates model architecture
  - Loads saved weights
  - Used for final test evaluation

#### **ALTERNATIVES**:
| What We Did | Alternative | Trade-off |
|-------------|-------------|-----------|
| T5-small (60M) | T5-base (220M) | Larger = better but slower |
| Full fine-tuning | Freeze encoder | Frozen = faster but less adaptive |
| AdamW optimizer | SGD | AdamW better for transformers |

---

### 3️⃣ **Evaluation Loop (train_t5.py - eval_epoch)**

#### **WHAT**: Evaluate model on dev set during training

#### **WHY**:
- Track progress (is model learning?)
- Decide when to stop (early stopping)
- Select best model (highest F1, not lowest loss!)

#### **HOW**:

```python
# For each batch:
1. Compute loss (with teacher forcing)
   - Forward pass with ground truth decoder inputs
   - Calculate cross-entropy loss
   - Track average loss

2. Generate SQL (with beam search)
   - model.generate() with num_beams=5
   - Explores 5 hypotheses in parallel
   - Better quality than greedy decoding

3. Execute SQL on database
   - save_queries_and_records()
   - Runs SQL queries on flight_database.db
   - Saves results to .pkl file

4. Compute metrics
   - Record F1: Main metric (how well do records match?)
   - Record EM: Exact match of records
   - SQL EM: Exact match of SQL strings
   - Error rate: % of queries with syntax errors
```

#### **Key Design Choices**:

**Beam Search (num_beams=5)**:
- **WHAT**: Keep top 5 hypotheses at each step
- **WHY**: Better quality than greedy (picking just the best token)
- **ALTERNATIVE**: Greedy decoding (faster but lower quality)

**Why Both Loss AND F1?**:
- **Loss**: Good training signal (differentiable)
- **F1**: Actual task performance (what we care about!)
- **Problem**: Loss and F1 not perfectly correlated
- **Solution**: Use F1 for model selection, loss for monitoring

#### **ALTERNATIVES**:
| What We Did | Alternative | Trade-off |
|-------------|-------------|-----------|
| Beam search (5) | Greedy decoding | Beam = better quality, slower |
| num_beams=5 | num_beams=10 | More beams = better but much slower |
| Execute on DB | String matching | Execution catches semantic errors |

---

### 4️⃣ **Test Inference (train_t5.py - test_inference)**

#### **WHAT**: Generate final predictions for test set

#### **WHY**:
- Test set has no labels (can't compute metrics)
- Just generate SQL and save for submission
- Use same settings as dev evaluation (consistency!)

#### **HOW**:

```python
# Almost same as eval_epoch but simpler:
1. No loss computation (no targets available)
2. No metrics computation (no ground truth)
3. Just generate SQL with beam search
4. Save to .sql and .pkl files
5. Submit to Gradescope!
```

---

## 🔧 How the Code Executes (Full Flow)

```
main() [train_t5.py]
│
├─ load_t5_data()
│   ├─ T5Dataset("train")
│   │   ├─ Load train.nl, train.sql
│   │   ├─ Tokenize: "text" → [token_ids]
│   │   └─ Create decoder inputs (shift by 1)
│   ├─ T5Dataset("dev")
│   └─ T5Dataset("test")
│
├─ initialize_model()
│   └─ Load T5-small with pretrained weights
│
├─ initialize_optimizer_and_scheduler()
│   ├─ AdamW optimizer (lr=5e-4)
│   └─ Cosine scheduler with warmup
│
├─ train()
│   └─ For each epoch:
│       ├─ train_epoch()
│       │   ├─ Forward pass: model(input, decoder_input)
│       │   ├─ Compute loss on decoder_targets
│       │   ├─ Backward pass: loss.backward()
│       │   └─ Update weights: optimizer.step()
│       │
│       ├─ eval_epoch()
│       │   ├─ Generate SQL with beam search
│       │   ├─ Execute on database
│       │   ├─ Compute F1, EM, error rate
│       │   └─ Save best model if F1 improved
│       │
│       └─ Check early stopping (patience=3)
│
└─ test_inference()
    ├─ Load best model
    ├─ Generate SQL for test set
    └─ Save to results/ and records/
```

---

## 🎓 Why This Baseline Works

### 1. **Pretrained Weights**
- T5 already knows English grammar, semantics
- Just needs to learn SQL syntax and schema
- Much easier than learning everything from scratch

### 2. **Task Prefix**
- "translate English to SQL: ..."
- T5 was pretrained with task descriptions
- Helps model understand what to do

### 3. **Teacher Forcing During Training**
- Decoder sees correct previous tokens (not its own predictions)
- Speeds up training significantly
- Standard practice for sequence-to-sequence models

### 4. **Beam Search During Inference**
- Explores multiple hypotheses
- Finds better SQL queries than greedy
- 5 beams is sweet spot (quality vs speed)

### 5. **Database Execution for Evaluation**
- Different SQL can produce same results
- String matching would miss semantically correct queries
- F1 on records is the right metric

---

## 🚀 How to Run the Baseline

### Command:
```bash
python train_t5.py \
    --finetune \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --batch_size 16 \
    --test_batch_size 32 \
    --max_n_epochs 20 \
    --patience_epochs 3 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --optimizer_type AdamW \
    --experiment_name baseline
```

### Expected Results:
- **Dev F1**: ~65-70% (should improve each epoch)
- **Test F1**: ≥65% (needed for full credit)
- **Training time**: ~2-3 hours on GPU
- **Each epoch**: ~8-10 minutes

---

## 🎨 What You Can Tune (For Better Performance)

### Easy Wins:
1. **Learning rate**: Try 1e-4 or 1e-3
2. **Batch size**: Try 32 if you have more GPU memory
3. **Num beams**: Try 10 for better quality (slower)
4. **Max epochs**: Train longer if still improving

### Advanced:
1. **Data augmentation**: Paraphrase NL queries
2. **Schema in input**: Add database schema to encoder input
3. **Constrained decoding**: Force valid SQL syntax
4. **Ensemble**: Average predictions from multiple models

---

## ⚠️ Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: Decrease batch_size to 8 or 4

### Issue: "Model not learning (F1 stays low)"
**Solution**:
- Check if loss is decreasing (if yes, keep training)
- Try higher learning rate (1e-3)
- Verify data loaded correctly (print some examples)

### Issue: "High error rate (syntax errors)"
**Solution**:
- Normal in early epochs
- Should decrease as training progresses
- Check examples to debug

### Issue: "F1 stops improving after few epochs"
**Solution**:
- This is normal! Early stopping will kick in
- Try lowering learning rate
- Or use learning rate schedule (we already do!)

---

## 📊 Understanding the Metrics

### **Record F1** (PRIMARY METRIC):
- F1 score between database records
- Example: Model returns 8/10 correct flights → F1 ≈ 0.8
- **WHY**: Handles partial matches gracefully

### **Record EM** (Exact Match):
- 1 if records exactly match, 0 otherwise
- Stricter than F1
- **WHY**: Good for debugging

### **SQL EM**:
- 1 if SQL strings exactly match
- Very strict (different SQL can be equivalent!)
- **WHY**: Useful but not main metric

### **Error Rate**:
- % of queries that caused SQL errors
- Should be low (<5%) for good model
- **WHY**: Identifies syntax issues

---

## ✅ Checklist Before Running

- [ ] Environment activated: `conda activate hw4-part-2-nlp`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] GPU available: Check `nvidia-smi` or `torch.cuda.is_available()`
- [ ] Data files present: `data/train.nl`, `data/train.sql`, etc.
- [ ] Directories exist: `results/`, `records/`, `checkpoints/`

---

## 📦 What Gets Saved

### During Training:
- `checkpoints/ft_experiments/baseline/best_model.pt` - Best model (highest dev F1)
- `checkpoints/ft_experiments/baseline/last_model.pt` - Most recent model
- `results/t5_ft_ft_experiment_dev.sql` - Generated SQL for dev set
- `records/t5_ft_ft_experiment_dev.pkl` - Database records for dev set

### For Submission:
- `results/t5_ft_ft_experiment_test.sql` - **SUBMIT THIS**
- `records/t5_ft_ft_experiment_test.pkl` - **SUBMIT THIS**

---

## 🎯 Expected Timeline

1. **Data Loading**: ~1-2 minutes (one-time at start)
2. **Epoch 1**: ~10 minutes (slowest, includes compilation)
3. **Subsequent epochs**: ~8 minutes each
4. **Early stopping**: Usually around epoch 10-15
5. **Total training**: 2-3 hours
6. **Test inference**: ~5 minutes

---

## 🧠 Key Takeaways

1. **Pretrained models are powerful**: Fine-tuning >> training from scratch
2. **Task formatting matters**: T5 needs task prefixes
3. **Evaluation is complex**: Can't just compare strings, need to execute SQL
4. **Beam search > Greedy**: Small cost, big quality improvement
5. **F1 > Loss**: Optimize for the metric you care about

Good luck with your training! 🚀
