package gin.edit.llm;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import org.checkerframework.checker.units.qual.s;

import dev.langchain4j.model.openai.OpenAiModelName;
import gin.edit.Edit;
import gin.edit.statement.DeleteStatement;
import gin.edit.statement.ReplaceStatement;
import gin.edit.statement.CopyStatement;
import gin.edit.statement.SwapStatement;
import gin.edit.llm.PromptTemplate.PromptTag;

public class LLMConfig {

	/** the following are some default template prompts */
	public enum PromptType {
		SIMPLE(new PromptTemplate("Give me " + PromptTag.COUNT.withEscape() + " implementations of this:"
        		+ "```\n"
        		+ PromptTag.DESTINATION.withEscape()
        		+ "\n"
        		+ "```\n")), 
		
		MEDIUM(new PromptTemplate("Give me " + PromptTag.COUNT.withEscape() + " different Java implementations of this method body:"
        		+ "```\n"
        		+ PromptTag.DESTINATION.withEscape()
        		+ "\n"
        		+ "```\n"
        		+ "This code belongs to project " + PromptTag.PROJECT.withEscape() + ". "
                + "Wrap all code in curly braces, if it is not already."
                + "Do not include any method or class declarations."
                + "label all code as java.")), 
		
		DETAILED(new PromptTemplate("Give me " + PromptTag.COUNT.withEscape() + " different Java implementations of this method body:"
        		+ "```\n"
        		+ PromptTag.DESTINATION.withEscape()
        		+ "\n"
        		+ "```\n"
        		+ "This code belongs to project " + PromptTag.PROJECT.withEscape() + ". "
        		+ "In the org.jcodec.scale.BaseResampler class, the following change was helpful. I changed this:"
        		+ "```\n"
        		+ "	if (temp == null) {"
        		+ "		temp = new int[toSize.getWidth() * (fromSize.getHeight() + nTaps())];"
        		+ "		tempBuffers.set(temp);"
        		+ "	}"
        		+ "```\n"
        		+ "into this:"
        		+ "```\n"
        		+ "	if (temp == null) {"
        		+ "		if (scaleFactorX >= 0)"
        		+ "			return;"
        		+ "		temp = new int[toSize.getWidth() * (fromSize.getHeight() + nTaps())];"
        		+ "		tempBuffers.set(temp);"
        		+ "	}"
        		+ "```\n"
                + "Wrap all code in curly braces, if it is not already."
                + "Do not include any method or class declarations."
                + "label all code as java.")), 


		MASKED(new PromptTemplate(
			  "You are optimizing a single Java statement to improve runtime while preserving correctness.\n"
			+ "\n"
			+ "Return exactly " + PromptTag.COUNT.withEscape() + " alternatives.\n"
			+ "For each alternative, output only one ```java code block containing either a single Java statement or a braced block that compiles at the placeholder site.\n"
			+ "Do not include any method or class declarations. Do not include any text outside the code blocks.\n"
			+ "\n"
			+ "Original statement:\n"
			+ "```\n"
			+ PromptTag.ORIGINAL_CODE.withEscape()
			+ "\n"
			+ "```\n"
			+ "Method body with placeholder (<<PLACEHOLDER>> is where your code must go):\n"
			+ "```java\n"
			+ PromptTag.DESTINATION.withEscape()
			+ "\n"
			+ "```\n"
			+ "\n"
			+ "Constraints:\n"
			+ "- Must compile in this context; do not add imports.\n"
			+ "- Use only variables shown above and class fields; no new global state.\n"
			+ "- Do not change method or class signatures; keep behavior and side-effects equivalent.\n"
			+ "- Prefer micro-optimizations: avoid allocations, avoid repeated work, hoist loop-invariant computations, use primitive operations, cache lengths, and short-circuit cheap checks.\n"
			+ "- Keep the replacement concise (ideally ≤ 3 lines).\n"
			+ "- Label all code blocks as java."
		)),

		MASKED_WITH_EXAMPLES(new PromptTemplate("Please replace <<PLACEHOLDER>> sign in the function below with meaningfull implementation. Please give " + PromptTag.COUNT.withEscape() + " different java implementation:\n"
				+ "\n"
				+ "```\n"
				+ PromptTag.DESTINATION.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "This code belongs to project " + PromptTag.PROJECT.withEscape() + ". Wrap all code in curly braces, if it is not already. Do not include any class declarations. Label all code as java.\n"
				+ "\n"
				+ "Here are some example input and output pairs:\n"
				+ "\n"
				+ "If you have input like this: \n"
				+ "\n"
				+ "```\n"
				+ "{\n"
				+ "    int q2 = pelsQ[q2Idx];\n"
				+ "    int diff = (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1;\n"
				+ "    diff = diff < -tC0 ? -tC0 : (diff > tC0 ? tC0 : diff);\n"
				+ "    int q1n = q1 + diff;\n"
				+ "    // <<PLACEHOLDER>>\n"
				+ "    ;\n"
				+ "}\n"
				+ "```\n"
				+ "\n"
				+ "Example output 1:\n"
				+ "\n"
				+ "```\n"
				+ "{\n"
				+ "    int q2 = pelsQ[q2Idx];\n"
				+ "    int diff = (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1;\n"
				+ "    diff = diff < -tC0 ? -tC0 : (diff > tC0 ? tC0 : diff);\n"
				+ "    int q1n = q1 + diff;\n"
				+ "    // <<PLACEHOLDER>> — clamp to [0..255] and write back q1\n"
				+ "    int q1c = q1n < 0 ? 0 : (q1n > 255 ? 255 : q1n);\n"
				+ "    pelsQ[q1Idx] = q1c;\n"
				+ "}\n"
				+ "```\n"
				+ "\n"
				+ "Example output 2:\n"
				+ "\n"
				+ "```\n"
				+ "{\n"
				+ "    int q2 = pelsQ[q2Idx];\n"
				+ "    int diff = (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1;\n"
				+ "    diff = diff < -tC0 ? -tC0 : (diff > tC0 ? tC0 : diff);\n"
				+ "    int q1n = q1 + diff;\n"
				+ "    // <<PLACEHOLDER>> — avoid writing for negligible changes\n"
				+ "    if (diff != 0) {\n"
				+ "        q1n = q1n < 0 ? 0 : (q1n > 255 ? 255 : q1n);\n"
				+ "        pelsQ[q1Idx] = q1n;\n"
				+ "    }\n"
				+ "}\n"
				+ "```\n")),

		MASKED_WITH_CONTEXT(new PromptTemplate("You are improving the performance of Java code by replacing a placeholder with optimized implementation. Please give " + PromptTag.COUNT.withEscape() + " different Java implementations.\n"
				+ "\n"
				+ "### Context Information\n"
				+ "\n"
				+ "**Project:** " + PromptTag.PROJECT.withEscape() + "\n"
				+ "\n"
				+ "**Method Signature:**\n"
				+ "```\n"
				+ PromptTag.METHOD_SIGNATURE.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "**Available Class Fields:**\n"
				+ "```\n"
				+ PromptTag.CLASS_FIELDS.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "**Imports:**\n"
				+ "```\n"
				+ PromptTag.IMPORTS.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "**Local Variables in Scope:**\n"
				+ "```\n"
				+ PromptTag.LOCAL_VARIABLES.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "### Method Body with Placeholder\n"
				+ "\n"
				+ "Please replace the <<PLACEHOLDER>> in the method below:\n"
				+ "\n"
				+ "```java\n"
				+ PromptTag.DESTINATION.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "### Requirements\n"
				+ "- Wrap all code in curly braces\n"
				+ "- Do not include any class or method declarations\n"
				+ "- Focus on performance optimization\n"
				+ "- Ensure the code compiles and maintains correctness\n"
				+ "- Label all code blocks as java\n"
				+ "\n"
				+ "### Output Format\n"
				+ "Provide " + PromptTag.COUNT.withEscape() + " different implementations, each in a separate ```java code block.\n"
				+ "Here are some example input and output pairs:\n"
				+ "\n"
				+ "If you have input like this: \n"
				+ "\n"
				+ "```\n"
				+ "{\n"
				+ "    int q2 = pelsQ[q2Idx];\n"
				+ "    int diff = (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1;\n"
				+ "    diff = diff < -tC0 ? -tC0 : (diff > tC0 ? tC0 : diff);\n"
				+ "    int q1n = q1 + diff;\n"
				+ "    // <<PLACEHOLDER>>\n"
				+ "    ;\n"
				+ "}\n"
				+ "```\n"
				+ "\n"
				+ "Example output 1:\n"
				+ "\n"
				+ "```\n"
				+ "{\n"
				+ "    int q2 = pelsQ[q2Idx];\n"
				+ "    int diff = (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1;\n"
				+ "    diff = diff < -tC0 ? -tC0 : (diff > tC0 ? tC0 : diff);\n"
				+ "    int q1n = q1 + diff;\n"
				+ "    // <<PLACEHOLDER>> — clamp to [0..255] and write back q1\n"
				+ "    int q1c = q1n < 0 ? 0 : (q1n > 255 ? 255 : q1n);\n"
				+ "    pelsQ[q1Idx] = q1c;\n"
				+ "}\n"
				+ "```\n"
				+ "\n"
				+ "Example output 2:\n"
				+ "\n"
				+ "```\n"
				+ "{\n"
				+ "    int q2 = pelsQ[q2Idx];\n"
				+ "    int diff = (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1;\n"
				+ "    diff = diff < -tC0 ? -tC0 : (diff > tC0 ? tC0 : diff);\n"
				+ "    int q1n = q1 + diff;\n"
				+ "    // <<PLACEHOLDER>> — avoid writing for negligible changes\n"
				+ "    if (diff != 0) {\n"
				+ "        q1n = q1n < 0 ? 0 : (q1n > 255 ? 255 : q1n);\n"
				+ "        pelsQ[q1Idx] = q1n;\n"
				+ "    }\n"
				+ "}\n"
				+ "```\n")),


		DELETION_TARGET_SELECTION(new PromptTemplate("You are improving the runtime of the Java method below by performing a **delete-mutation** that must **compile and keep (or increase) the current test-suite pass rate**. \n"
				+ "### Your task"
				+ "\n"
				+ "1. Pick **exactly one** statement whose removal is unlikely to break compilation or logic (e.g. redundant work inside a deep loop, dead code, or a side-effect-free update). \n"
				+ "2. Removing it should *ideally* reduce execution time without reducing the number of passing tests. \n"
				+ "3. **Respond with only the numeric ID** of that statement — no extra text, punctuation, or whitespace. \n"
				+ "\n"
				+ "**Constraints** \n"
				+ "* Do **not** delete variable or method declarations that later code depends on. \n"
				+ "* Prefer deletions inside nested loops or obviously duplicated calculations. \n"
				+ "* The resulting code must remain syntactically valid Java. \n"
				+ "\n"
				+ "### AST (ID → code)"
				+ "```\n"
				+ PromptTag.AST.withEscape()
				+ "\n"
				+ "```\n"
				+ "*End of input. Remember: output **only** the chosen ID.*")),
				
		REPLACE_TARGET_DESTINATION_SELECTION(new PromptTemplate("You are improving the runtime of the Java method below by performing a **replace-mutation** that must **compile and keep (or increase) the current test-suite pass rate**. \n"
				+ "### Your task"
				+ "\n"
				+ "1. Pick **exactly one** *target* statement to be replaced. \n"
				+ "2. Pick **exactly one** *destination* statement whose code will replace the target. \n"
				+ "3. Replacing it should *ideally* reduce execution time without reducing the number of passing tests. \n"
				+ "4. **Respond with only two numeric IDs separated by a single space**: `<targetID> <destinationID>` — no extra text, punctuation, or whitespace. \n"
				+ "\n"
				+ "**Constraints** \n"
				+ "* Do **not** choose variable or method declarations whose alteration would break later code. \n"
				+ "* The destination statement must be type-compatible with the target location. \n"
				+ "* The resulting code must remain syntactically valid Java. \n"
				+ "\n"
				+ "### AST (ID → code)"
				+ "```\n"
				+ PromptTag.AST.withEscape()
				+ "\n"
				+ "```\n"
				+ "*End of input. Remember: output **only** the two IDs.*")),

		COPY_TARGET_DESTINATION_SELECTION(new PromptTemplate(
					"You are improving the runtime of the Java method below by performing a **copy-mutation** that must **compile** and **keep (or increase) the current test-suite pass rate**.\n"
				  + "\n"
				  + "### Your task\n"
				  + "1. Pick **exactly one** *source* statement to copy.  \n"
				  + "2. Pick **exactly one** *destination* block (a `BlockStmt` that can contain statements).  \n"
				  + "3. Within that block, pick **exactly one** *destination child* statement **that is not itself a block**.  \n"
				  + "   - The copied code will be inserted **immediately before** the first statement whose ID is **greater than** this child ID.  \n"
				  + "4. **Respond with only three numeric IDs separated by single spaces**:  \n"
				  + "   \n"
				  + "   `<sourceID> <destinationBlockID> <destinationChildID>`\n"
				  + "   \n"
				  + "   — no extra text, punctuation, or whitespace.\n"
				  + "\n"
				  + "### Constraints\n"
				  + "* Avoid copying declarations thateditToPromptTemplate would cause duplicate definitions.  \n"
				  + "* Do **not** copy control-flow headers (`if`, `for`, etc.) or braces alone.  \n"
				  + "* Ensure the insertion point keeps the code syntactically valid and semantically sound.  \n"
				  + "* The child ID **must not** refer to a block; it must be a regular statement within `destinationBlock`.\n"
				  + "\n"
				  + "### AST (ID → code)\n"
				  + "```\\n"
				  + PromptTag.AST.withEscape()
				  + "\\n```\n"
				  + "*End of input. Remember: output **only** the three IDs.*"
			)),
			

		SWAP_TARGET_DESTINATION_SELECTION(new PromptTemplate("You are improving the runtime of the Java method below by performing a **swap-mutation** that must **compile and keep (or increase) the current test-suite pass rate**. \n"
				+ "### Your task"
				+ "\n"
				+ "1. Pick **exactly two** statements whose order can be safely exchanged. \n"
				+ "2. Swapping them should *ideally* reduce execution time without reducing the number of passing tests. \n"
				+ "3. **Respond with only two numeric IDs separated by a single space** in the order they appear in the original code: `<firstID> <secondID>` — no extra text, punctuation, or whitespace. \n"
				+ "\n"
				+ "**Constraints** \n"
				+ "* Do **not** swap declarations that must precede their usage. \n"
				+ "* The resulting code must remain syntactically valid Java. \n"
				+ "\n"
				+ "### AST (ID → code)"
				+ "```\n"
				+ PromptTag.AST.withEscape()
				+ "\n"
				+ "```\n"
				+ "*End of input. Remember: output **only** the two IDs.*")),

		MASK_DESTINATION_SELECTION(new PromptTemplate(
				  "You are selecting a single Java statement to be replaced by an optimized implementation that improves runtime while preserving correctness.\n"
				+ "\n"
				+ "### Your task\n"
				+ "1. Pick **exactly one** statement whose replacement is most likely to reduce execution time (e.g., inner-loop arithmetic, repeated computations, unnecessary allocations, repeated method calls, string concatenations).\n"
				+ "2. That statement will be replaced at the same location with a single statement or a small braced block.\n"
				+ "3. **Respond with only the numeric ID** of that statement — no extra text, punctuation, or whitespace.\n"
				+ "\n"
				+ "### Constraints\n"
				+ "- Do **not** pick variable or method declarations that later code depends on.\n"
				+ "- Avoid control-flow headers (`if`, `for`, `while`, `switch`, `try`) and `return`/`throw`/`break`/`continue` as targets.\n"
				+ "- Prefer statements inside performance-critical loops or obvious hot-spots and statements that are self-contained.\n"
				+ "- The code must remain syntactically valid Java after replacement.\n"
				+ "\n"
				+ "### Method body\n"
				+ "```java\n"
				+ PromptTag.DESTINATION.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "### Statement List (ID → code)\n"
				+ "```\n"
				+ PromptTag.AST.withEscape()
				+ "\n"
				+ "```\n"
				+ "*End of input. Remember: output **only** the chosen ID.*"
			)),


		// 	DELETION_TARGET_SELECTION_WITH_CODE(new PromptTemplate("You are improving the runtime of the Java method below by performing a **delete-mutation** that must **compile and keep (or increase) the current test-suite pass rate**. \n"
		// 		+ "### Your task"
		// 		+ "\n"
		// 		+ "1. Pick **exactly one** statement whose removal is unlikely to break compilation or logic (e.g. redundant work inside a deep loop, dead code, or a side-effect-free update). \n"
		// 		+ "2. Removing it should *ideally* reduce execution time without reducing the number of passing tests. \n"
		// 		+ "3. **Respond with only the numeric ID** of that statement — no extra text, punctuation, or whitespace. \n"
		// 		+ "\n"
		// 		+ "**Constraints** \n"
		// 		+ "* Do **not** delete variable or method declarations that later code depends on. \n"
		// 		+ "* Prefer deletions inside nested loops or obviously duplicated calculations. \n"
		// 		+ "* The resulting code must remain syntactically valid Java. \n"
		// 		+ "### Method code \n"
		// 		+ "```java\n"
		// 		+ PromptTag.DESTINATION.withEscape()
		// 		+ "\n"
		// 		+ "```\n"
		// 		+ "\n"
		// 		+ "### Statement List (ID → code)"
		// 		+ "```\n"
		// 		+ PromptTag.AST.withEscape()
		// 		+ "\n"
		// 		+ "```\n"
		// 		+ "*End of input. Remember: output **only** the chosen ID.*")),
				
		// REPLACE_TARGET_DESTINATION_SELECTION_WITH_CODE(new PromptTemplate("You are improving the runtime of the Java method below by performing a **replace-mutation** that must **compile and keep (or increase) the current test-suite pass rate**. \n"
		// 		+ "### Your task"
		// 		+ "\n"
		// 		+ "1. Pick **exactly one** *target* statement to be replaced. \n"
		// 		+ "2. Pick **exactly one** *destination* statement whose code will replace the target. \n"
		// 		+ "3. Replacing it should *ideally* reduce execution time without reducing the number of passing tests. \n"
		// 		+ "4. **Respond with only two numeric IDs separated by a single space**: `<targetID> <destinationID>` — no extra text, punctuation, or whitespace. \n"
		// 		+ "\n"
		// 		+ "**Constraints** \n"
		// 		+ "* Do **not** choose variable or method declarations whose alteration would break later code. \n"
		// 		+ "* The destination statement must be type-compatible with the target location. \n"
		// 		+ "* The resulting code must remain syntactically valid Java. \n"
		// 		+ "### Method code \n"
		// 		+ "```java\n"
		// 		+ PromptTag.DESTINATION.withEscape()
		// 		+ "\n"
		// 		+ "```\n"
		// 		+ "\n"
		// 		+ "### Statement List (ID → code)"
		// 		+ "```\n"
		// 		+ PromptTag.AST.withEscape()
		// 		+ "\n"
		// 		+ "```\n"
		// 		+ "*End of input. Remember: output **only** the two IDs.*")),

		// COPY_TARGET_DESTINATION_SELECTION_WITH_CODE(new PromptTemplate(
		// 			"You are improving the runtime of the Java method below by performing a **copy-mutation** that must **compile** and **keep (or increase) the current test-suite pass rate**.\n"
		// 		  + "\n"
		// 		  + "### Your task\n"
		// 		  + "1. Pick **exactly one** *source* statement to copy.  \n"
		// 		  + "2. Pick **exactly one** *destination* block (a `BlockStmt` that can contain statements).  \n"
		// 		  + "3. Within that block, pick **exactly one** *destination child* statement **that is not itself a block**.  \n"
		// 		  + "   - The copied code will be inserted **immediately before** the first statement whose ID is **greater than** this child ID.  \n"
		// 		  + "4. **Respond with only three numeric IDs separated by single spaces**:  \n"
		// 		  + "   \n"
		// 		  + "   `<sourceID> <destinationBlockID> <destinationChildID>`\n"
		// 		  + "   \n"
		// 		  + "   — no extra text, punctuation, or whitespace.\n"
		// 		  + "\n"
		// 		  + "### Constraints\n"
		// 		  + "* Avoid copying declarations thateditToPromptTemplate would cause duplicate definitions.  \n"
		// 		  + "* Do **not** copy control-flow headers (`if`, `for`, etc.) or braces alone.  \n"
		// 		  + "* Ensure the insertion point keeps the code syntactically valid and semantically sound.  \n"
		// 		  + "* The child ID **must not** refer to a block; it must be a regular statement within `destinationBlock`.\n"
		// 		  + "\n"
		// 		  + "### Method code \n"
		// 		  + "```java\n"
		// 		  + PromptTag.DESTINATION.withEscape()
		// 		  + "\n"
		// 		  + "```\n"
		// 		  + "\n"
		// 		  + "### Statement List (ID → code)\n"
		// 		  + "```\\n"
		// 		  + PromptTag.AST.withEscape()
		// 		  + "\\n```\n"
		// 		  + "*End of input. Remember: output **only** the three IDs.*"
		// 	)),
			

		// SWAP_TARGET_DESTINATION_SELECTION_WITH_CODE(new PromptTemplate("You are improving the runtime of the Java method below by performing a **swap-mutation** that must **compile and keep (or increase) the current test-suite pass rate**. \n"
		// 		+ "### Your task"
		// 		+ "\n"
		// 		+ "1. Pick **exactly two** statements whose order can be safely exchanged. \n"
		// 		+ "2. Swapping them should *ideally* reduce execution time without reducing the number of passing tests. \n"
		// 		+ "3. **Respond with only two numeric IDs separated by a single space** in the order they appear in the original code: `<firstID> <secondID>` — no extra text, punctuation, or whitespace. \n"
		// 		+ "\n"
		// 		+ "**Constraints** \n"
		// 		+ "* Do **not** swap declarations that must precede their usage. \n"
		// 		+ "* The resulting code must remain syntactically valid Java. \n"
		// 		+ "### Method code \n"
		// 		+ "```java\n"
		// 		+ PromptTag.DESTINATION.withEscape()
		// 		+ "\n"
		// 		+ "```\n"
		// 		+ "\n"
		// 		+ "### Statement List (ID → code)"
		// 		+ "```\n"
		// 		+ PromptTag.AST.withEscape()
		// 		+ "\n"
		// 		+ "```\n"
		// 		+ "*End of input. Remember: output **only** the two IDs.*")),

		// MASK_DESTINATION_SELECTION_WITH_CODE(new PromptTemplate("You are improving the runtime of the Java method below by performing a **mask-mutation** that will later be rewritten by another LLM call.\n"
		// 		+ "\n"
		// 		+ "### Your task\n"
		// 		+ "1. Pick **exactly one** statement whose replacement is most likely to reduce execution time (e.g., inner-loop arithmetic, redundant computations, unnecessary allocations).  \n"
		// 		+ "2. The chosen statement will be temporarily replaced with a placeholder so new code can be generated in its place.  \n"
		// 		+ "3. **Respond with only the numeric ID** of that statement — no extra text, punctuation, or whitespace.  \n"
		// 		+ "\n"
		// 		+ "**Constraints**  \n"
		// 		+ "* Do **not** pick variable or method declarations that later code depends on.  \n"
		// 		+ "* Prefer statements inside performance-critical loops or obvious hot-spots.  \n"
		// 		+ "* The resulting code must remain syntactically valid Java after the placeholder is inserted.  \n"
		// 		+ "\n"
		// 		+ "### Method code \n"
		// 		+ "```java\n"
		// 		+ PromptTag.DESTINATION.withEscape()
		// 		+ "\n"
		// 		+ "```\n"
		// 		+ "\n"
		// 		+ "### Statement List (ID → code)\n"
		// 		+ "```\\n"
		// 		+ PromptTag.AST.withEscape()
		// 		+ "\\n```\n"
		// 		+ "*End of input. Remember: output **only** the chosen ID.*"
		// 	)),
		


		DELETION_TARGET_SELECTION_WITH_CODE(new PromptTemplate("You are an expert Java optimization engine. Perform a **delete-mutation** to improve runtime while maintaining the test pass rate. \n"
				+ "### Your task"
				+ "\n"
				+ "1. Identify **exactly one** statement that is redundant, dead code, or a side-effect-free calculation whose result is unused. \n"
				+ "2. Removing it must not break the logic of the program. \n"
				+ "3. Provide a brief **reasoning** explaining why this statement is safe to delete. \n"
				+ "4. Identify the **numeric ID** of that statement. \n"
				+ "\n"
				+ "**Constraints** \n"
				+ "* **Dependency Check**: Do **not** delete a variable definition (e.g., `int x = ...`) if `x` is used in any subsequent statement. \n"
				+ "* **Control Flow**: Do not delete control headers (like `if` or `while`) unless you intend to delete the flow structure entirely (which is risky). \n"
				+ "* The resulting code must remain syntactically valid Java. \n"
				+ "\n"
				+ "### JSON output (respond with json only)\n"
				+ "Example:\n"
				+ "```json\n"
				+ "{\n"
				+ "  \"reasoning\": \"Brief reason here\",\n"
				+ "  \"StatementID\": 123\n"
				+ "}\n"
				+ "```\n"
				+ "### Method code \n"
				+ "```java\n"
				+ PromptTag.DESTINATION.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "### Statement List (ID → code)"
				+ "```\n"
				+ PromptTag.AST.withEscape()
				+ "\n"
				+ "```\n"
				+ "*End of input. Return json only.*")),
				
		REPLACE_TARGET_DESTINATION_SELECTION_WITH_CODE(new PromptTemplate("You are an expert Java optimization engine. Perform a **replace-mutation** to improve runtime. \n"
				+ "### Your task"
				+ "\n"
				+ "1. Pick **exactly one** *target* statement (the location to be overwritten). \n"
				+ "2. Pick **exactly one** *ingredient* statement (the code source to copy). \n"
				+ "3. The *ingredient* code will replace the *target* code completely. \n"
				+ "4. Provide a brief **reasoning** explaining why this replacement is valid and beneficial. \n"
				+ "5. Identify the **targetID** and **ingredientID**. \n"
				+ "\n"
				+ "**Constraints** \n"
				+ "* **Scope Safety**: You must ensure that every variable used in the *ingredient* statement is defined and valid at the *target* location. \n"
				+ "* **Type Compatibility**: The *ingredient* must make sense in the *target* context (e.g., do not replace a statement inside a loop with a variable declaration from outside that loop). \n"
				+ "* Do **not** overwrite variable declarations that are relied upon by later code. \n"
				+ "\n"
				+ "### JSON output (respond with json only)\n"
				+ "Example:\n"
				+ "```json\n"
				+ "{\n"
				+ "  \"reasoning\": \"Brief reason here\",\n"
				+ "  \"TargetID\": 12,\n"
				+ "  \"ingredientID\": 34\n"
				+ "}\n"
				+ "```\n"
				+ "### Method code \n"
				+ "```java\n"
				+ PromptTag.DESTINATION.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "### Statement List (ID → code)"
				+ "```\n"
				+ PromptTag.AST.withEscape()
				+ "\n"
				+ "```\n"
				+ "*End of input. Return json only.*")),

		COPY_TARGET_DESTINATION_SELECTION_WITH_CODE(new PromptTemplate(
					"You are an expert Java optimization engine. Perform a **copy-mutation** to improve runtime. \n"
				  + "\n"
				  + "### Your task\n"
				  + "1. Pick **exactly one** *ingredient* statement (code to reuse).  \n"
				  + "2. Pick **exactly one** *target block* (a `BlockStmt` or loop body where code can be inserted).  \n"
				  + "3. Within that block, pick **exactly one** *anchor* statement. The ingredient will be inserted **immediately before** this anchor.  \n"
				  + "4. Provide a brief **reasoning** explaining why this copy operation is valid and beneficial. \n"
				  + "5. Identify the **ingredientID**, **targetBlockID**, and **anchorID**. \n"
				  + "\n"
				  + "### Constraints\n"
				  + "* **Scope Safety**: The *ingredient* code must only use variables that are available inside the *target block*.  \n"
				  + "* **Duplication**: Do not insert a variable declaration if that variable is already defined in the *target block* (this causes compile errors).  \n"
				  + "* The *anchor* must be a direct child of the *target block*. \n"
				  + "\n"
				  + "### JSON output (respond with json only)\n"
				  + "Example:\n"
				  + "```json\n"
				  + "{\n"
				  + "  \"reasoning\": \"Brief reason here\",\n"
				  + "  \"ingredientID\": 7,\n"
				  + "  \"targetBlockID\": 21,\n"
				  + "  \"anchorID\": 25\n"
				  + "}\n"
				  + "```\n"
				  + "\n"
				  + "### Method code \n"
				  + "```java\n"
				  + PromptTag.DESTINATION.withEscape()
				  + "\n"
				  + "```\n"
				  + "\n"
				  + "### Statement List (ID → code)\n"
				  + "```\\n"
				  + PromptTag.AST.withEscape()
				  + "\\n```\n"
				  + "*End of input. Return json only.*"
			)),
			

		SWAP_TARGET_DESTINATION_SELECTION_WITH_CODE(new PromptTemplate("You are an expert Java optimization engine. Perform a **swap-mutation** to improve runtime. \n"
				+ "### Your task"
				+ "\n"
				+ "1. Pick **exactly two** statements whose order can be exchanged to improve performance (e.g., moving a cheap check before an expensive calculation). \n"
				+ "2. Provide a brief **reasoning** explaining why swapping these statements preserves correctness and improves performance. \n"
				+ "3. Identify the **firstID** and **secondID**. \n"
				+ "\n"
				+ "**Constraints** \n"
				+ "* **Data Dependency**: Do not swap Statement A and Statement B if B uses a variable defined in A. Breaking definition-usage order will fail compilation. \n"
				+ "* **Scope**: Do not swap statements across different scopes (e.g., inside vs outside a loop) unless variable availability permits it. \n"
				+ "* The resulting code must remain syntactically valid Java. \n"
				+ "\n"
				+ "### JSON output (respond with json only)\n"
				+ "Example:\n"
				+ "```json\n"
				+ "{\n"
				+ "  \"reasoning\": \"Brief reason here\",\n"
				+ "  \"firstID\": 5,\n"
				+ "  \"secondID\": 9\n"
				+ "}\n"
				+ "```\n"
				+ "### Method code \n"
				+ "```java\n"
				+ PromptTag.DESTINATION.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "### Statement List (ID → code)"
				+ "```\n"
				+ PromptTag.AST.withEscape()
				+ "\n"
				+ "```\n"
				+ "*End of input. Return json only.*")),

		MASK_DESTINATION_SELECTION_WITH_CODE(new PromptTemplate("You are an expert Java optimization engine. Perform a **mask-mutation** to identify inefficient code. \n"
				+ "\n"
				+ "### Your task\n"
				+ "1. Identify **exactly one** 'hotspot' statement that is computationally expensive (e.g., inner-loop arithmetic, complex method calls, or redundant allocations).  \n"
				+ "2. This statement will be temporarily removed and rewritten by an AI generator.  \n"
				+ "3. Provide a brief **reasoning** explaining why this statement is a good candidate for optimization. \n"
				+ "4. Identify the **numeric ID** of that statement. \n"
				+ "\n"
				+ "**Constraints**  \n"
				+ "* Do **not** mask variable declarations if the variable is used by later code (unless the whole block is being rewritten).  \n"
				+ "* Focus on the deepest loops or most frequently executed lines.  \n"
				+ "* The resulting code must remain syntactically valid Java.  \n"
				+ "\n"
				+ "### JSON output (respond with json only)\n"
				+ "Example:\n"
				+ "```json\n"
				+ "{\n"
				+ "  \"reasoning\": \"Brief reason here\",\n"
				+ "  \"StatementID\": 42\n"
				+ "}\n"
				+ "```\n"
				+ "\n"
				+ "### Method code \n"
				+ "```java\n"
				+ PromptTag.DESTINATION.withEscape()
				+ "\n"
				+ "```\n"
				+ "\n"
				+ "### Statement List (ID → code)\n"
				+ "```\\n"
				+ PromptTag.AST.withEscape()
				+ "\\n```\n"
				+ "*End of input. Return json only.*"
			)),

	WITH_MUTATION_EXAMPLES(new PromptTemplate("Please give me " + PromptTag.COUNT.withEscape() + " different Java implementations of this method body:\n"
			  + "\n"
			  + "```\n"
			  + PromptTag.DESTINATION.withEscape()
			  + "\n"
			  + "```\n"
			  + "\n"
			  + "This code belongs to project " + PromptTag.PROJECT.withEscape() + ".\n"
			  + "\n"
			  + "Wrap all code in curly braces, if it is not already. Do not include any method or class declarations. Label all code as java.\n"
			  + "\n"
			  + "I will give you four examples of small changes you could try. If you have this code:\n"
			  + "\n"
			  + "```\n"
			  + "    public void write(ByteBuffer buf) {\n"
			  + "        ByteBuffer dup = buf.duplicate();\n"
			  + "        NIOUtils.skip(buf, 8);\n"
			  + "        doWrite(buf);\n"
			  + "        header.setBodySize(buf.position() - dup.position() - 8);\n"
			  + "        Assert.assertEquals(header.headerSize(), 8);\n"
			  + "        header.write(dup);\n"
			  + "    }\n"
			  + "```\n"
			  + "\n"
			  + "Example 1: you could try copying a statement from one place to another like this\n"
			  + "\n"
			  + "```\n"
			  + "    public void write(ByteBuffer buf) {\n"
			  + "        ByteBuffer dup = buf.duplicate();\n"
			  + "        NIOUtils.skip(buf, 8);\n"
			  + "        doWrite(buf);\n"
			  + "        header.setBodySize(buf.position() - dup.position() - 8);\n"
			  + "        Assert.assertEquals(header.headerSize(), 8);\n"
			  + "        Assert.assertEquals(header.headerSize(), 8);\n"
			  + "        header.write(dup);\n"
			  + "    }\n"
			  + "```\n"
			  + "\n"
			  + "Example 2: you could try deleting a statement chosen at random like this:\n"
			  + "\n"
			  + "```\n"
			  + "    public void write(ByteBuffer buf) {\n"
			  + "        ByteBuffer dup = buf.duplicate();\n"
			  + "        NIOUtils.skip(buf, 8);\n"
			  + "        header.setBodySize(buf.position() - dup.position() - 8);\n"
			  + "        Assert.assertEquals(header.headerSize(), 8);\n"
			  + "        header.write(dup);\n"
			  + "    }\n"
			  + "```\n"
			  + "\n"
			  + "Example 3: you could try replacing one statement with another like this:\n"
			  + "\n"
			  + "```\n"
			  + "    public void write(ByteBuffer buf) {\n"
			  + "        ByteBuffer dup = buf.duplicate();\n"
			  + "        NIOUtils.skip(buf, 8);\n"
			  + "        doWrite(buf);\n"
			  + "        header.setBodySize(buf.position() - dup.position() - 8);\n"
			  + "        Assert.assertEquals(header.headerSize(), 8);\n"
			  + "        NIOUtils.skip(buf, 8);\n"
			  + "    }\n"
			  + "```\n"
			  + "\n"
			  + "Example 4: you could try swapping two statements like this:\n"
			  + "\n"
			  + "```\n"
			  + "    public void write(ByteBuffer buf) {\n"
			  + "        ByteBuffer dup = buf.duplicate();\n"
			  + "        doWrite(buf);\n"
			  + "        NIOUtils.skip(buf, 8);\n"
			  + "        header.setBodySize(buf.position() - dup.position() - 8);\n"
			  + "        Assert.assertEquals(header.headerSize(), 8);\n"
			  + "        NIOUtils.skip(buf, 8);\n"
			  + "    }\n"
			  + "```\n"
			  + "\n"
			  + "In all of these examples, the statements to change are chosen at random. They do not have to be whole lines, just valid Java statements.\n"
		)),

	RUNTIME_OPTIMISATION(new PromptTemplate("You are an expert Java performance optimization engineer. Please give me " + PromptTag.COUNT.withEscape() + " different optimized Java implementations of this method body that improve runtime performance:\n"
			  + "\n"
			  + "```java\n"
			  + PromptTag.DESTINATION.withEscape()
			  + "\n"
			  + "```\n"
			  + "\n"
			  + "This code belongs to project " + PromptTag.PROJECT.withEscape() + ".\n"
			  + "\n"
			  + "### Optimization Goal\n"
			  + "Reduce the **runtime/execution time** of this method while **preserving its exact functional behavior**. The optimized code must produce the same outputs for all possible inputs.\n"
			  + "\n"
			  + "### Optimization Techniques to Consider\n"
			  + "- **Algorithmic improvements**: Use more efficient algorithms or data structures\n"
			  + "- **Loop optimizations**: Reduce iterations, use early exits, loop unrolling, or eliminate redundant computations inside loops\n"
			  + "- **Caching/memoization**: Cache repeated computations or method call results\n"
			  + "- **Avoid unnecessary object allocations**: Reuse objects, use primitives instead of wrappers\n"
			  + "- **Reduce method calls**: Inline small methods, avoid unnecessary calls\n"
			  + "- **Use efficient operations**: Prefer bitwise operations, avoid expensive operations like division when possible\n"
			  + "- **Short-circuit evaluation**: Reorder conditions to fail fast\n"
			  + "- **Eliminate dead code**: Remove code that does not affect the output\n"
			  + "\n"
			  + "### Constraints\n"
			  + "- **DO NOT** change the functional behavior or output of the code\n"
			  + "- **DO NOT** introduce any new bugs or break existing functionality\n"
			  + "- The optimized code must remain syntactically valid Java\n"
			  + "- Wrap all code in curly braces, if it is not already\n"
			  + "- Do not include any method or class declarations\n"
			  + "- Label all code blocks as java\n"
			  + "\n"
			  + "### Output Format\n"
			  + "Provide " + PromptTag.COUNT.withEscape() + " different optimized implementations, each in a separate ```java code block. For each implementation, briefly explain the optimization technique used.\n"
			  + "\n"
			  + "Example output format:\n"
			  + "```java\n"
			  + "{\n"
			  + "    try {\n"
			  + "        return dateFormat.parse(s);\n"
			  + "    }\n"
			  + "    catch (ParseException ignored) {\n"
			  + "    }\n"
			  + "}\n"
			  + "```\n"
		)),

	RUNTIME_OPTIMISATION_WITH_EXAMPLES(new PromptTemplate("You are an expert Java performance optimization engineer. Please give me " + PromptTag.COUNT.withEscape() + " different optimized Java implementations of this method body that improve runtime performance:\n"
			  + "\n"
			  + "```java\n"
			  + PromptTag.DESTINATION.withEscape()
			  + "\n"
			  + "```\n"
			  + "\n"
			  + "This code belongs to project " + PromptTag.PROJECT.withEscape() + ".\n"
			  + "\n"
			  + "### Optimization Goal\n"
			  + "Reduce the **runtime/execution time** of this method while **preserving its exact functional behavior**. The optimized code must produce the same outputs for all possible inputs.\n"
			  + "\n"
			  + "### Optimization Techniques to Consider\n"
			  + "- **Algorithmic improvements**: Use more efficient algorithms or data structures\n"
			  + "- **Loop optimizations**: Reduce iterations, use early exits, loop unrolling, or eliminate redundant computations inside loops\n"
			  + "- **Caching/memoization**: Cache repeated computations or method call results\n"
			  + "- **Avoid unnecessary object allocations**: Reuse objects, use primitives instead of wrappers\n"
			  + "- **Reduce method calls**: Inline small methods, avoid unnecessary calls\n"
			  + "- **Use efficient operations**: Prefer bitwise operations, avoid expensive operations like division when possible\n"
			  + "- **Short-circuit evaluation**: Reorder conditions to fail fast\n"
			  + "- **Eliminate dead code**: Remove code that does not affect the output\n"
			  + "\n"
			  + "### Examples\n"
			  + "\n"
			  + "**Example 1: Loop-invariant code motion (hoisting)**\n"
			  + "\n"
			  + "Original code:\n"
			  + "```java\n"
			  + "{\n"
			  + "    int sum = 0;\n"
			  + "    for (int i = 0; i < arr.length; i++) {\n"
			  + "        int multiplier = config.getMultiplier();\n"
			  + "        sum += arr[i] * multiplier;\n"
			  + "    }\n"
			  + "    return sum;\n"
			  + "}\n"
			  + "```\n"
			  + "\n"
			  + "Optimized code:\n"
			  + "```java\n"
			  + "{\n"
			  + "    int sum = 0;\n"
			  + "    int multiplier = config.getMultiplier();\n"
			  + "    for (int i = 0; i < arr.length; i++) {\n"
			  + "        sum += arr[i] * multiplier;\n"
			  + "    }\n"
			  + "    return sum;\n"
			  + "}\n"
			  + "```\n"
			  + "Optimization: Moved the invariant method call outside the loop to avoid redundant calls.\n"
			  + "\n"
			  + "**Example 2: Early exit optimization**\n"
			  + "\n"
			  + "Original code:\n"
			  + "```java\n"
			  + "{\n"
			  + "    boolean found = false;\n"
			  + "    for (int i = 0; i < list.size(); i++) {\n"
			  + "        if (list.get(i).equals(target)) {\n"
			  + "            found = true;\n"
			  + "        }\n"
			  + "    }\n"
			  + "    return found;\n"
			  + "}\n"
			  + "```\n"
			  + "\n"
			  + "Optimized code:\n"
			  + "```java\n"
			  + "{\n"
			  + "    for (int i = 0; i < list.size(); i++) {\n"
			  + "        if (list.get(i).equals(target)) {\n"
			  + "            return true;\n"
			  + "        }\n"
			  + "    }\n"
			  + "    return false;\n"
			  + "}\n"
			  + "```\n"
			  + "Optimization: Added early return to exit immediately when target is found.\n"
			  + "\n"
			  + "**Example 3: Avoiding redundant object allocations**\n"
			  + "\n"
			  + "Original code:\n"
			  + "```java\n"
			  + "{\n"
			  + "    String result = \"\";\n"
			  + "    for (String s : items) {\n"
			  + "        result = result + s + \",\";\n"
			  + "    }\n"
			  + "    return result;\n"
			  + "}\n"
			  + "```\n"
			  + "\n"
			  + "Optimized code:\n"
			  + "```java\n"
			  + "{\n"
			  + "    StringBuilder sb = new StringBuilder();\n"
			  + "    for (String s : items) {\n"
			  + "        sb.append(s).append(\",\");\n"
			  + "    }\n"
			  + "    return sb.toString();\n"
			  + "}\n"
			  + "```\n"
			  + "Optimization: Used StringBuilder to avoid creating multiple intermediate String objects.\n"
			  + "\n"
			  + "### Constraints\n"
			  + "- **DO NOT** change the functional behavior or output of the code\n"
			  + "- **DO NOT** introduce any new bugs or break existing functionality\n"
			  + "- The optimized code must remain syntactically valid Java\n"
			  + "- Wrap all code in curly braces, if it is not already\n"
			  + "- Do not include any method or class declarations\n"
			  + "- Label all code blocks as java\n"
			  + "\n"
			  + "### Output Format\n"
			  + "Provide " + PromptTag.COUNT.withEscape() + " different optimized implementations, each in a separate ```java code block. For each implementation, briefly explain the optimization technique used.\n"
			  + "\n"
			  + "Example output format:\n"
			  + "```java\n"
			  + "{\n"
			  + "    try {\n"
			  + "        return dateFormat.parse(s);\n"
			  + "    }\n"
			  + "    catch (ParseException ignored) {\n"
			  + "    }\n"
			  + "}\n"
			  + "```\n"
		)),;
		
		
		
		public final PromptTemplate template;
	    private PromptType(PromptTemplate template) {
	        this.template = template;
	    }
	}

	public enum FormatType {
		SINGLE_STATEMENT_ID(createSingleStatementFormat()),
		TWO_STATEMENT_IDS_REPLACEMENT(createTwoStatementFormat_Replacement()),
		TWO_STATEMENT_IDS_SWAP(createTwoStatementFormat_Swap()),
		THREE_STATEMENT_IDS(createThreeStatementFormat());

		public final Map<String, Object> format;
		
		private FormatType(Map<String, Object> format) {
			this.format = format;
		}

		/**
		 * Creates format for single statement ID: {"StatementID": integer}
		 */
		private static Map<String, Object> createSingleStatementFormat() {
			Map<String, Object> reasoning = new HashMap<>();
			reasoning.put("type", "string");
			reasoning.put("description", "Explanation for why this statement was chosen");

			Map<String, Object> statementID = new HashMap<>();
			statementID.put("type", Integer.class.getSimpleName().toLowerCase());

			Map<String, Object> properties = new LinkedHashMap<>();
			properties.put("reasoning", reasoning);
			properties.put("StatementID", statementID);

			Map<String, Object> format = new LinkedHashMap<>();
			format.put("type", "object");
			format.put("properties", properties);
			format.put("required", Arrays.asList("reasoning", "StatementID"));
			
			return format;
		}

		/**
		 * Creates format for two statement IDs: {"TargetID": integer, "DestinationID": integer}
		 */
		private static Map<String, Object> createTwoStatementFormat_Replacement() {
			Map<String, Object> reasoning = new HashMap<>();
			reasoning.put("type", "string");
			reasoning.put("description", "Explanation for why this replacement was chosen");

			Map<String, Object> targetID = new HashMap<>();
			targetID.put("type", Integer.class.getSimpleName().toLowerCase());

			Map<String, Object> destinationID = new HashMap<>();
			destinationID.put("type", Integer.class.getSimpleName().toLowerCase());

			Map<String, Object> properties = new LinkedHashMap<>();
			properties.put("reasoning", reasoning);
			properties.put("TargetID", targetID);
			properties.put("ingredientID", destinationID);

			Map<String, Object> format = new LinkedHashMap<>();
			format.put("type", "object");
			format.put("properties", properties);
			format.put("required", Arrays.asList("reasoning", "TargetID", "ingredientID"));
			
			return format;
		}

		private static Map<String, Object> createTwoStatementFormat_Swap() {
			Map<String, Object> reasoning = new HashMap<>();
			reasoning.put("type", "string");
			reasoning.put("description", "Explanation for why these statements were swapped");

			Map<String, Object> firstID = new HashMap<>();
			firstID.put("type", Integer.class.getSimpleName().toLowerCase());

			Map<String, Object> secondID = new HashMap<>();
			secondID.put("type", Integer.class.getSimpleName().toLowerCase());

			Map<String, Object> properties = new LinkedHashMap<>();
			properties.put("reasoning", reasoning);
			properties.put("firstID", firstID);
			properties.put("secondID", secondID);

			Map<String, Object> format = new LinkedHashMap<>();
			format.put("type", "object");
			format.put("properties", properties);
			format.put("required", Arrays.asList("reasoning", "firstID", "secondID"));
			
			return format;
		}

		/**
		 * Creates format for three statement IDs: 
		 * {"SourceID": integer, "DestinationBlockID": integer, "DestinationChildID": integer}
		 */
		private static Map<String, Object> createThreeStatementFormat() {
			Map<String, Object> reasoning = new HashMap<>();
			reasoning.put("type", "string");
			reasoning.put("description", "Explanation for why this copy operation was chosen");

			Map<String, Object> sourceID = new HashMap<>();
			sourceID.put("type", Integer.class.getSimpleName().toLowerCase());

			Map<String, Object> destinationBlockID = new HashMap<>();
			destinationBlockID.put("type", Integer.class.getSimpleName().toLowerCase());

			Map<String, Object> destinationChildID = new HashMap<>();
			destinationChildID.put("type", Integer.class.getSimpleName().toLowerCase());

			Map<String, Object> properties = new LinkedHashMap<>();
			properties.put("reasoning", reasoning);
			properties.put("ingredientID", sourceID);
			properties.put("targetBlockID", destinationBlockID);
			properties.put("anchorID", destinationChildID);

			Map<String, Object> format = new LinkedHashMap<>();
			format.put("type", "object");
			format.put("properties", properties);
			format.put("required", Arrays.asList("reasoning", "ingredientID", "targetBlockID", "anchorID"));
			
			return format;
		}
	}

	private static final Map<Class<? extends Edit>, PromptTemplate> editToPromptTemplate = new HashMap<>();

	// static {
	// 	editToPromptTemplate.put(DeleteStatement.class, PromptType.DELETION_TARGET_SELECTION.template);
	// 	editToPromptTemplate.put(ReplaceStatement.class, PromptType.REPLACE_TARGET_DESTINATION_SELECTION.template);
	// 	editToPromptTemplate.put(CopyStatement.class, PromptType.COPY_TARGET_DESTINATION_SELECTION.template);
	// 	editToPromptTemplate.put(SwapStatement.class, PromptType.SWAP_TARGET_DESTINATION_SELECTION.template);
	// 	editToPromptTemplate.put(LLMMaskedStatement.class, PromptType.MASK_DESTINATION_SELECTION.template);
	// }

	static {
		editToPromptTemplate.put(DeleteStatement.class, PromptType.DELETION_TARGET_SELECTION_WITH_CODE.template);
		editToPromptTemplate.put(ReplaceStatement.class, PromptType.REPLACE_TARGET_DESTINATION_SELECTION_WITH_CODE.template);
		editToPromptTemplate.put(CopyStatement.class, PromptType.COPY_TARGET_DESTINATION_SELECTION_WITH_CODE.template);
		editToPromptTemplate.put(SwapStatement.class, PromptType.SWAP_TARGET_DESTINATION_SELECTION_WITH_CODE.template);
		editToPromptTemplate.put(LLMMaskedStatement.class, PromptType.MASK_DESTINATION_SELECTION_WITH_CODE.template);
	}

	public static PromptTemplate getPromptTemplateForEdit(Class<? extends Edit> editClass) {
		return editToPromptTemplate.get(editClass);
	}

	private static final Map<Class<? extends Edit>, Map<String, Object>> editToFormat = new HashMap<>();

	static {
		// Single statement ID: DeleteStatement, LLMMaskedStatement
		editToFormat.put(DeleteStatement.class, FormatType.SINGLE_STATEMENT_ID.format);
		editToFormat.put(LLMMaskedStatement.class, FormatType.SINGLE_STATEMENT_ID.format);
		
		// Two statement IDs: ReplaceStatement, SwapStatement
		editToFormat.put(ReplaceStatement.class, FormatType.TWO_STATEMENT_IDS_REPLACEMENT.format);
		editToFormat.put(SwapStatement.class, FormatType.TWO_STATEMENT_IDS_SWAP.format);
		
		// Three statement IDs: CopyStatement
		editToFormat.put(CopyStatement.class, FormatType.THREE_STATEMENT_IDS.format);
	}

	public static Map<String, Object> getFormatForEdit(Class<? extends Edit> editClass) {
		return editToFormat.get(editClass);
	}
	
	// You can use "demo" api key for demonstration purposes.
	public static String openAIKey = "demo";

	public static String openAIModelName = "gpt-5-nano";

	public static String deepSeekKey = "demo";

	public static String deepSeekModelName = "deepseek-v4-pro";

	public static String deepSeekBaseUrl = "https://api.deepseek.com";

	/** Enable DeepSeek thinking mode. Both deepseek-v4-pro and deepseek-v4-flash
	 * support a "thinking" object on the request body with type=enabled|disabled.
	 * When enabled, the model produces a reasoning_content field alongside content. */
	public static boolean deepSeekThinking = false;

	/** DeepSeek reasoning effort when thinking is enabled. Accepts "high" or "max".
	 * Legacy "low"/"medium" map to "high"; "xhigh" maps to "max". */
	public static String deepSeekReasoningEffort = "max";

	public static String modelType="OpenAI"; // Should be param from c'tor

    public static long timeoutInSeconds = 30;
    
    // default for langchain4j
    public static double temperature = 0.7;
    
    public static PromptType defaultPromptType = PromptType.MEDIUM;
    
    public static PromptTemplate defaultPromptTemplate = null;
    
    public static PromptTemplate getDefaultPromptTemplate() {
    	return (defaultPromptTemplate != null) ? defaultPromptTemplate : defaultPromptType.template;
    }
    
    public static String projectName = "";

	public static boolean isOpenAICompatibleModelType() {
		return "OpenAI".equalsIgnoreCase(modelType) || "DeepSeek".equalsIgnoreCase(modelType);
	}
    
    
    
}