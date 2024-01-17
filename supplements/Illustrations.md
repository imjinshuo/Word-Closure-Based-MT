# Illustrations of the Motivation Examples

## SIT-FN
**SUT**: Google Translate

**S_s**: Steve Kennedy, the **`treasury`** secretary, forced it to alert the people who have bought a phone found to be substandard, rather than **<font color=red>taking down listing</font>**.

**S_f**: Steve Kennedy, the **`interior`** secretary, forced it to alert the people who have bought a phone found to be substandard, rather than **<font color=red>taking down listing</font>**.

**T_s**: <font color=blue>*美国*</font>`财政`部长史蒂夫·肯尼迪强迫该公司向那些购买了不合格手机的人发出警告，而不是<font color=red>*下架*</font>。

**Meaning of T_s**: Steve Kennedy, the treasury secretary **<font color=blue>*of the U.S.*</font>,** forced it to alert the people who have bought a phone found to be substandard, rather than **<font color=red>*removed from shelves*</font>**.

**T_f**: `内政`部长史蒂夫·肯尼迪强迫该公司向那些购买了不合格手机的人发出警告，而不是<font color=red>*删除清单*</font>。

**Meaning of T_f**: Steve Kennedy, the interior secretary, forced it to alert the people who have bought a phone found to be substandard, rather than **<font color=red>*delete list*</font>**.

**Explanation:** 
- **Why this example is a violation:**
  - The input words "taking down listing" are translated as different meanings in T_s and T_f, i.e., "removed from shelves" in T_s and "delete list" in T_f.
  - T_s have a redundant word "美国", which means "the U.S." and does not exist in T_f.
- **Why this example is identified as non-violation by SIT:**
  - The dependency parse tree of T_s:
  - <img src="../figs/motivationExample-SIT(1).png" height = "250">
  - The dependency parse tree of T_f:
  - <img src="../figs/motivationExample-SIT(2).png" height = "250">
  - Distance between the tags of the two dependency parse trees is 2, which is below the threshold (6), thus SIT regards this example as non-violation.

## CAT-FP
**SUT**: Google Translate

**S_s**: The app will be released next `Sunday` **<font color=blue>when everything is ready</font>**.

**S_f**: The app will be released next `year` **<font color=blue>when everything is ready</font>**.

**T_s**: 该应用程序将于下`周日`<font color=blue>*一切准备就绪后*</font>发布。

**Meaning of T_s**: The app will be released next Sunday **<font color=blue>*when everything is ready*</font>**.

**T_f**: <font color=blue>*当一切准备就绪后*</font>，该应用程序将于明`年`发布。

**Meaning of T_f**: **<font color=blue>*When everything is ready*</font>**, the app will be released next Sunday.

**Explanation:**
- Why this example is not a violation:
  - Although the adverbial clause of "当一切准备就绪后" (when everything is ready) is at a different position in T_s and T_f, it does not change meaning of the sentence.
- Why this example is identified as a violation by CAT:
  - Similarity score (calculated by CAT) between T_s and T_f, which is 37.8%, is below the threshold (80.0%), thus CAT reports this example as a violation.

## Purity-FP
**SUT**: Google Translate

**S_s**: English **<font color=blue>tests</font>** and **<font color=red>Chinese tests</font>** for the **<font color=green>children</font>** `will be canceled on Tuesday.`

**S_f**: English **<font color=blue>tests</font>** and **<font color=red>Chinese tests</font>** for the **<font color=green>children</font>**

**T_s**: `周二`<font color=green>*孩子们*</font>的英语<font color=blue>*考试*</font>和<font color=red>*中文考试*</font>`将被取消`。

**Meaning of T_s**: English **<font color=blue>*tests*</font>** and **<font color=red>*Chinese tests*</font>** for the **<font color=green>*children*</font>** will be canceled on Tuesday.

**T_f**: <font color=green>*儿童*</font>英语<font color=blue>*测试*</font>和<font color=red>*汉语测试*</font>

**Meaning of T_f**: English **<font color=blue>*tests*</font>** and **<font color=red>*Chinese tests*</font>** for the **<font color=green>*children*</font>**

**Explanation:**
- **Why this example is not a violation:**
  - "孩子们" in T_s and "儿童" in T_f share the same meaning, i.e., "children".
  - "考试" in T_s and "测试" in T_f share the same meaning, i.e., "tests".
  - "中文考试" in T_s and "汉语测试" in T_f share the same meaning, i.e., "Chinese tests".
- **Why this example is identified as a violation by Purity:**
  - Four words from T_f do not exist in T_s. As a result, the distance between T_s and T_f is 4, which is above the threshold of Purity (1), thus Purity reports this example as a violation.

## CIT-FN
**SUT**: Bing Microsoft Translator

**S_s**: The small **<font color=red>assorted</font>** sesame is easier to sell in the market. 

**S_f**: `Therefore,` the small **<font color=red>assorted</font>** sesame is easier to sell in the market.

**T_s**: 小芝麻在市场上更容易销售。

**Meaning of T_s**: The small sesame is easier to sell in the market. 

**T_f**: `因此，`小<font color=red>*什锦*</font>芝麻更容易在市场上销售。

**Meaning of T_f**: Therefore, the small **<font color=red>*assorted*</font>** sesame is easier to sell in the market.

**Explanation:** 
- **Why this example is a violation:**
  - The input word "assorted" is not translated in T_s but is translated in T_f as "什锦" (assorted).
- **Why this example is identified as non-violation by CIT:**
  - The constituency parse tree of T_s:
  - <img src="../figs/motivationExample-CIT(1).png" height = "200">
  - The constituency parse tree of T_f:
  - <img src="../figs/motivationExample-CIT(2).png" height = "200">
  - Every path in the constituency parse tree of T_s exists in that of T_f, thus CIT regards this example as non-violation.

## PatInv-FN
**SUT**: Bing Microsoft Translator

**S_s**: In January, most **<font color=red>policies</font>** would offer the maintenance costs of the building during the `pandemic`. 

**S_f**: In January, most **<font color=red>policies</font>** would offer the maintenance costs of the building during the `holiday`.

**T_s**: 在 1 月份，大多数<font color=red>*保单*</font>都会提供`大流行`期间建筑物的维护成本。

**Meaning of T_s**: In January, most **<font color=red>*guarantee slip*</font>** would offer the maintenance costs of the building during the pandemic. 

**T_f**: 在一月份，大多数<font color=red>*政策*</font>都会在`假期`期间提供建筑物的维护费用。

**Meaning of T_f**: In January, most **<font color=red>*policies*</font>** would offer the maintenance costs of the building during the holiday.

**Explanation:** 
- **Why this example is a violation:**
  - The input word "policies" is translated as different meanings, i.e., "保单" (guarantee slip) in T_s and "政策" (policies) in T_f.
- **Why this example is identified as a violation by PatInv:**
  - T_s and T_f are different, thus PatInv regards this example as non-violation.