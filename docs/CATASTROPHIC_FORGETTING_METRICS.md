# Catastrophic Forgetting Metrics: Equations and Extensions

## Current Implementation Equations

### 1. Absolute Degradation (Current)
For task $i$ at training step $t$:

$$D_i^t = acc_i^0 - acc_i^t$$

Where:
- $acc_i^0$ = accuracy of task $i$ at baseline (step 0)
- $acc_i^t$ = accuracy of task $i$ at step $t$
- $D_i^t$ = absolute degradation (positive = forgetting, negative = improvement)

**Interpretation**: Raw difference in performance from baseline. Higher positive values indicate more forgetting.

### 2. Relative Degradation (Current)
For task $i$ at training step $t$:

$$RD_i^t = \frac{acc_i^0 - acc_i^t}{acc_i^0} \times 100\%$$

**Interpretation**: Percentage change relative to baseline. Useful for comparing tasks with different baseline performances.

### 3. Degradation Rate (Current)
At training step $t$:

$$DR^t = \frac{|\{i : acc_i^t < acc_i^0\}|}{N} \times 100\%$$

Where $N$ = total number of tasks.

**Interpretation**: Percentage of tasks showing any degradation from baseline.

### 4. Average Degradation (Current)
At training step $t$:

$$\bar{D}^t = \frac{1}{N} \sum_{i=1}^{N} D_i^t = \frac{1}{N} \sum_{i=1}^{N} (acc_i^0 - acc_i^t)$$

**Interpretation**: Mean degradation across all tasks.

---

## Additional Metrics (Recommended)

### 5. **Forgetting Measure (F)** - Lopez-Paz & Ranzato (2017)
More sophisticated measure that considers the best performance achieved:

$$F_i^t = \max_{k \in \{0, \ldots, t-1\}} acc_i^k - acc_i^t$$

**Interpretation**: Measures forgetting relative to the best performance ever achieved, not just baseline. This captures recovery scenarios.

**Advantage**: Accounts for cases where performance improves then degrades.

### 6. **Backward Transfer (BWT)**
Measures how learning new tasks affects previously learned tasks:

$$BWT^t = \frac{1}{N-1} \sum_{i=1}^{N-1} (acc_i^t - acc_i^{t-1})$$

Or more commonly, the average change from baseline:

$$BWT^t = \frac{1}{N} \sum_{i=1}^{N} (acc_i^t - acc_i^0)$$

**Interpretation**: Positive = positive transfer, Negative = negative transfer (forgetting).

### 7. **Forward Transfer (FWT)**
Measures how previous learning helps with new tasks:

$$FWT^t = \frac{1}{M} \sum_{j=1}^{M} (acc_j^t - acc_j^{random})$$

Where $M$ = number of new tasks at step $t$, and $acc_j^{random}$ = random baseline.

**Interpretation**: How much better the model performs on new tasks compared to random chance.

### 8. **Retention Rate**
Percentage of original performance retained:

$$RR_i^t = \frac{acc_i^t}{acc_i^0} \times 100\%$$

**Interpretation**: 
- 100% = perfect retention
- < 100% = forgetting occurred
- > 100% = improvement over baseline

### 9. **Stability-Plasticity Trade-off Index**
Measures the balance between maintaining old knowledge and learning new:

$$SP^t = \frac{\text{Avg Retention}}{\text{Avg New Task Performance}} = \frac{\frac{1}{N} \sum_{i=1}^{N} RR_i^t}{\frac{1}{M} \sum_{j=1}^{M} acc_j^t}$$

**Interpretation**: Higher values indicate better balance.

### 10. **Area Under the Learning Curve (AULC)**
Cumulative performance over training:

$$AULC_i = \sum_{t=0}^{T} acc_i^t \times \Delta t$$

Where $\Delta t$ = step interval.

**Interpretation**: Overall learning efficiency - higher is better.

### 11. **Forgetting Rate (Temporal)**
Rate of change in forgetting:

$$FR_i^t = \frac{D_i^t - D_i^{t-1}}{\Delta t}$$

**Interpretation**: How quickly forgetting is occurring (acceleration of forgetting).

### 12. **Recovery Index**
Measures ability to recover from forgetting:

$$RI_i^t = \frac{acc_i^t - \min_{k \in \{0, \ldots, t\}} acc_i^k}{\max_{k \in \{0, \ldots, t\}} acc_i^k - \min_{k \in \{0, \ldots, t\}} acc_i^k}$$

**Interpretation**: 
- 0 = at worst performance
- 1 = at best performance
- Values between indicate recovery progress

### 13. **Task Difficulty-Adjusted Forgetting**
Normalizes forgetting by task difficulty:

$$DAF_i^t = \frac{D_i^t}{1 - acc_i^0}$$

**Interpretation**: Accounts for ceiling effects - tasks already near perfect have less room to forget.

### 14. **Catastrophic Forgetting Score (CFS)**
Composite metric combining multiple factors:

$$CFS^t = w_1 \cdot \bar{D}^t + w_2 \cdot DR^t + w_3 \cdot \max_i D_i^t$$

Where $w_1, w_2, w_3$ are weights (e.g., 0.4, 0.3, 0.3).

**Interpretation**: Single score summarizing overall forgetting severity.

### 15. **Task Similarity-Weighted Forgetting**
Forgetting weighted by task similarity (if task embeddings available):

$$SWF_i^t = \sum_{j \neq i} \text{sim}(i,j) \cdot D_j^t$$

Where $\text{sim}(i,j)$ = similarity between tasks $i$ and $j$.

**Interpretation**: Measures forgetting in related tasks, which may indicate transfer effects.

---

## Recommended Implementation Priority

### High Priority (Most Useful)
1. **Forgetting Measure (F)** - More accurate than simple degradation
2. **Retention Rate** - Intuitive and easy to interpret
3. **Backward Transfer (BWT)** - Standard in continual learning literature

### Medium Priority
4. **Recovery Index** - Useful if performance fluctuates
5. **Area Under Learning Curve** - Good for overall assessment
6. **Task Difficulty-Adjusted Forgetting** - Accounts for ceiling effects

### Low Priority (Research/Advanced)
7. **Forward Transfer** - Requires identifying "new" tasks
8. **Stability-Plasticity Index** - Requires task categorization
9. **Task Similarity-Weighted** - Requires task embeddings

---

## Example Calculation

For task "global_mmlu_full_pl_college_computer_science":
- Baseline (step 0): $acc_0 = 0.37$
- Final (step 3080): $acc_{3080} = 0.29$

**Current Metrics:**
- Absolute Degradation: $D = 0.37 - 0.29 = 0.08$
- Relative Degradation: $RD = \frac{0.08}{0.37} \times 100\% = 21.62\%$
- Retention Rate: $RR = \frac{0.29}{0.37} \times 100\% = 78.38\%$

**If we had intermediate steps:**
- Forgetting Measure (F): Would track max performance across all steps
- Recovery Index: Would show if performance recovered at any point

---

## References

1. **Lopez-Paz, D., & Ranzato, M.** (2017). Gradient episodic memory for continual learning. *NeurIPS*.
2. **Kirkpatrick, J., et al.** (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
3. **Chaudhry, A., et al.** (2018). Riemannian walk for incremental learning. *ECCV*.

