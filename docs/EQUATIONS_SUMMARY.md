# Catastrophic Forgetting Equations Summary

## Current Implementation

### 1. Absolute Degradation
$$D_i^t = acc_i^0 - acc_i^t$$

**Where:**
- $acc_i^0$ = accuracy of task $i$ at baseline (step 0)
- $acc_i^t$ = accuracy of task $i$ at step $t$
- $D_i^t$ = absolute degradation

**Interpretation:** 
- Positive values = forgetting occurred
- Negative values = improvement over baseline
- Zero = no change

**Example:** If baseline accuracy is 0.37 and current is 0.29, then $D = 0.37 - 0.29 = 0.08$ (8% absolute drop)

---

### 2. Relative Degradation
$$RD_i^t = \frac{acc_i^0 - acc_i^t}{acc_i^0} \times 100\%$$

**Interpretation:**
- Percentage change relative to baseline
- Useful for comparing tasks with different baseline performances
- 0% = no change, positive % = forgetting, negative % = improvement

**Example:** With baseline 0.37 and current 0.29: $RD = \frac{0.08}{0.37} \times 100\% = 21.62\%$

---

### 3. Retention Rate (NEW)
$$RR_i^t = \frac{acc_i^t}{acc_i^0} \times 100\%$$

**Interpretation:**
- Percentage of original performance retained
- 100% = perfect retention
- < 100% = forgetting occurred
- > 100% = improvement over baseline

**Example:** With baseline 0.37 and current 0.29: $RR = \frac{0.29}{0.37} \times 100\% = 78.38\%$

---

### 4. Forgetting Measure (NEW) - Lopez-Paz & Ranzato (2017)
$$F_i^t = \max_{k \in \{0, \ldots, t-1\}} acc_i^k - acc_i^t$$

**Interpretation:**
- Measures forgetting relative to the **best performance ever achieved**, not just baseline
- More accurate than simple degradation when performance improves then degrades
- Captures recovery scenarios

**Example:** If best performance was 0.40 at step 1000, and current is 0.29, then $F = 0.40 - 0.29 = 0.11$

---

### 5. Backward Transfer (NEW)
$$BWT_i^t = acc_i^t - acc_i^0$$

**Interpretation:**
- Measures how learning new tasks affects previously learned tasks
- Positive = positive transfer (improvement)
- Negative = negative transfer (forgetting)
- Zero = no transfer effect

**Example:** With baseline 0.37 and current 0.29: $BWT = 0.29 - 0.37 = -0.08$ (negative transfer)

---

### 6. Degradation Rate (Aggregate)
$$DR^t = \frac{|\{i : acc_i^t < acc_i^0\}|}{N} \times 100\%$$

**Where:**
- $N$ = total number of tasks
- $|\{i : acc_i^t < acc_i^0\}|$ = count of tasks showing degradation

**Interpretation:** Percentage of tasks showing any degradation from baseline

---

### 7. Average Degradation (Aggregate)
$$\bar{D}^t = \frac{1}{N} \sum_{i=1}^{N} D_i^t = \frac{1}{N} \sum_{i=1}^{N} (acc_i^0 - acc_i^t)$$

**Interpretation:** Mean degradation across all tasks at step $t$

---

## Relationship Between Metrics

- **Absolute Degradation** and **Backward Transfer** are opposites:
  $$D_i^t = -BWT_i^t$$

- **Retention Rate** and **Relative Degradation** are related:
  $$RR_i^t = 100\% - RD_i^t$$

- **Forgetting Measure** is always $\geq$ **Absolute Degradation**:
  $$F_i^t \geq D_i^t$$
  (because max performance $\geq$ baseline performance)

---

## When to Use Which Metric?

| Metric | Best For | Limitation |
|--------|----------|-----------|
| **Absolute Degradation** | Simple comparison, raw numbers | Doesn't account for baseline differences |
| **Relative Degradation** | Comparing tasks with different baselines | Can be misleading for very low baselines |
| **Retention Rate** | Intuitive interpretation | Same as relative degradation (inverted) |
| **Forgetting Measure** | Tracking recovery scenarios | More complex to compute |
| **Backward Transfer** | Measuring transfer effects | Same as absolute degradation (inverted) |

---

## Implementation Status

✅ **Implemented:**
- Absolute Degradation
- Relative Degradation
- Retention Rate
- Forgetting Measure
- Backward Transfer
- Degradation Rate
- Average Degradation

📋 **Available for Future Implementation:**
- Forward Transfer (requires identifying "new" tasks)
- Area Under Learning Curve
- Recovery Index
- Task Difficulty-Adjusted Forgetting
- Stability-Plasticity Trade-off Index

