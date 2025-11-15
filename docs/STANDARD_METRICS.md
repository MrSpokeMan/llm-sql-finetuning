# Standard Catastrophic Forgetting Metrics

## Implemented Metrics

### 1. Forgetting Measure (FORG)
**Definition:** Measures the decrease in performance on a previous task after training on new tasks. Calculated as the difference between the best accuracy on a task and the accuracy after learning subsequent tasks.

**Equation:**
$$FORG_i^t = \max_{k \in \{0, \ldots, t-1\}} acc_i^k - acc_i^t$$

**Where:**
- $acc_i^k$ = accuracy of task $i$ at step $k$
- $acc_i^t$ = accuracy of task $i$ at current step $t$

**Interpretation:**
- Positive values = forgetting occurred (performance dropped from best)
- Zero = no forgetting (at best performance)
- Negative values = improvement (exceeded previous best)

**Implementation:** `forgetting_measure` in `CatastrophicForgettingMetrics`, `avg_forgetting_measure` in `StepSummary`

---

### 2. Average Accuracy (ACC)
**Definition:** Measures overall retained performance across all tasks after sequential training.

**Equation:**
$$ACC^t = \frac{1}{N} \sum_{i=1}^{N} acc_i^t$$

**Where:**
- $N$ = total number of tasks
- $acc_i^t$ = accuracy of task $i$ at step $t$

**Interpretation:**
- Higher values = better overall performance
- Measures how well the model retains knowledge across all tasks

**Implementation:** `avg_performance` in `StepSummary`

---

### 3. Backward Transfer (BWT)
**Definition:** Measures how learning new tasks affects the performance on old tasks negatively. Catastrophic forgetting manifests as negative BWT.

**Equation:**
$$BWT_i^t = acc_i^t - acc_i^0$$

**Where:**
- $acc_i^0$ = baseline accuracy (before learning new tasks)
- $acc_i^t$ = accuracy after learning new tasks

**Interpretation:**
- Positive BWT = positive transfer (improvement)
- Negative BWT = negative transfer (forgetting) ← **This is catastrophic forgetting**
- Zero = no transfer effect

**Implementation:** `backward_transfer` in `CatastrophicForgettingMetrics`, `avg_backward_transfer` in `StepSummary`

---

### 4. Catastrophic Forgetting Test (CBT)
**Definition:** The average difference between the performance of the last observed task and all previous tasks, quantifying how much knowledge is lost after learning new tasks.

**Equation:**
$$CBT^t = \frac{1}{N} \sum_{i=1}^{N} FORG_i^t = \frac{1}{N} \sum_{i=1}^{N} (\max_{k < t} acc_i^k - acc_i^t)$$

**Where:**
- $N$ = total number of tasks
- $FORG_i^t$ = forgetting measure for task $i$ at step $t$

**Interpretation:**
- Higher positive values = more catastrophic forgetting
- Zero = no forgetting (all tasks at their best performance)
- Measures the average amount of knowledge lost across all tasks

**Implementation:** `catastrophic_forgetting_test` in `StepSummary`

---

## Relationship Between Metrics

- **FORG** and **CBT**: CBT is the average of FORG across all tasks
  $$CBT^t = \frac{1}{N} \sum_{i=1}^{N} FORG_i^t$$

- **BWT** and **FORG**: 
  - If best performance = baseline: $FORG_i^t = -BWT_i^t$
  - If best performance > baseline: $FORG_i^t \geq -BWT_i^t$

- **ACC**: Independent measure of overall performance retention

---

## Usage in Analysis

### Example Report Structure
```json
{
  "step_summaries": {
    "3080": {
      "avg_performance": 0.2549,           // ACC
      "avg_backward_transfer": -0.0001,    // BWT (negative = forgetting)
      "avg_forgetting_measure": 0.0012,    // FORG
      "catastrophic_forgetting_test": 0.0012 // CBT
    }
  }
}
```

### Interpreting Results

**Good Performance (No Forgetting):**
- ACC: High (close to baseline)
- BWT: Close to zero or positive
- FORG: Close to zero
- CBT: Close to zero

**Catastrophic Forgetting:**
- ACC: Lower than baseline
- BWT: Negative (significant negative values)
- FORG: Positive (significant positive values)
- CBT: Positive (higher = worse)

---

## References

1. **Lopez-Paz, D., & Ranzato, M.** (2017). Gradient episodic memory for continual learning. *NeurIPS*.
2. **Chaudhry, A., et al.** (2018). Riemannian walk for incremental learning. *ECCV*.
3. **Kirkpatrick, J., et al.** (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.

