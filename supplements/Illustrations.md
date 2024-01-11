# Illustrations of the Motivation Examples

## SIT-FN
**SUT**: Google Translate

**S_s**: Steve Kennedy, the **`treasury`** secretary, forced it to alert the people who have bought a phone found to be substandard, rather than **<font color=red>taking down listing</font>**.

**S_f**: Steve Kennedy, the **`interior`** secretary, forced it to alert the people who have bought a phone found to be substandard, rather than **<font color=red>taking down listing</font>**.

**T_s**: <font color=red>*美国*</font>财政部长史蒂夫·肯尼迪强迫该公司向那些购买了不合格手机的人发出警告，而不是<font color=red>*下架*</font>。

**Meaning**: Steve Kennedy, the treasury secretary **<font color=red>*of the U.S.*</font>,** forced it to alert the people who have bought a phone found to be substandard, rather than **<font color=red>*removed from shelves*</font>**.

**T_f**: 内政部长史蒂夫·肯尼迪强迫该公司向那些购买了不合格手机的人发出警告，而不是<font color=red>*删除清单*</font>。

**Meaning**: Steve Kennedy, the treasury secretary, forced it to alert the people who have bought a phone found to be substandard, rather than **<font color=red>*delete list*</font>**.

**Explanation:** 
- The input words "taking down listing" are translated as different meanings in T_s and T_f, i.e., "removed from shelves" in T_s and "delete list" in T_f.
- T_s have a redundant word "美国", which means "the U.S." and does not exist in T_f.

## CAT-FP
**SUT**: Google Translate

**S_s**: The app will be released next `Sunday` **<font color=blue>when everything is ready</font>**.

**S_f**: The app will be released next `year` **<font color=blue>when everything is ready</font>**.

**T_s**: 该应用程序将于下周日<font color=blue>*一切准备就绪后发布*</font>。

**Meaning**: The app will be released next Sunday **<font color=blue>*when everything is ready*</font>**.

**T_f**: <font color=blue>*当一切准备就绪后*</font>，该应用程序将于明年发布。

**Meaning**: **<font color=blue>*When everything is ready*</font>**, the app will be released next Sunday.

**Explanation:**
- T_s and T_f share the same meaning.

## Purity-FP
**SUT**: Google Translate

**S_s**: English **<font color=blue>tests</font>** and **<font color=blue>Chinese tests</font>** for the **<font color=blue>children</font>** `will be canceled on Tuesday.`

**S_f**: English **<font color=blue>tests</font>** and **<font color=blue>Chinese tests</font>** for the **<font color=blue>children</font>**

**T_s**: 周二<font color=blue>*孩子们*</font>的英语<font color=blue>*考试*</font>和<font color=blue>*中文考试*</font>将被取消。

**Meaning**: English **<font color=blue>*tests*</font>** and **<font color=blue>*Chinese tests*</font>** for the **<font color=blue>*children*</font>** will be canceled on Tuesday.

**T_f**: <font color=blue>*儿童*</font>英语<font color=blue>*测试*</font>和<font color=blue>*汉语测试*</font>

**Meaning**: English **<font color=blue>*tests*</font>** and **<font color=blue>*Chinese tests*</font>** for the **<font color=blue>*children*</font>**

**Explanation:**
- "孩子们" in T_s and "儿童" in T_f share the same meaning, which are "children".
- "考试" in T_s and "测试" in T_f share the same meaning, which are "tests".
- "中文考试" in T_s and "汉语测试" in T_f share the same meaning, which are "Chinese tests".

## CIT-FN
**SUT**: Bing Microsoft Translator

**S_s**: The small **<font color=red>assorted</font>** sesame is easier to sell in the market. 

**S_f**: `Therefore,` the small **<font color=red>assorted</font>** sesame is easier to sell in the market.

**T_s**: 小芝麻在市场上更容易销售。

**Meaning**: The small sesame is easier to sell in the market. 

**T_f**: 因此，小<font color=red>*什锦*</font>芝麻更容易在市场上销售。

**Meaning**: Therefore, the small **<font color=red>*assorted*</font>** sesame is easier to sell in the market.

**Explanation:** 
- The input word "assorted" is not translated in T_s but is translated in T_f as "什锦".

## PatInv-FN
**SUT**: Bing Microsoft Translator

**S_s**: In January, most **<font color=red>policies</font>** would offer the maintenance costs of the building during the `pandemic`. 

**S_f**: In January, most **<font color=red>policies</font>** would offer the maintenance costs of the building during the `holiday`.

**T_s**: 在 1 月份，大多数<font color=red>*保单*</font>都会提供大流行期间建筑物的维护成本。

**Meaning**: In January, most **<font color=red>*guarantee slip*</font>** would offer the maintenance costs of the building during the pandemic. 

**T_f**: 在一月份，大多数<font color=red>*政策*</font>都会在假期期间提供建筑物的维护费用。

**Meaning**: In January, most **<font color=red>*policies*</font>** would offer the maintenance costs of the building during the holiday.

**Explanation:** 
- The input word "policies" is translated as different meanings, i.e., "guarantee slip" in T_s and "policies" in T_f.