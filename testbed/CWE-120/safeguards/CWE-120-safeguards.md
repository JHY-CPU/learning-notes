# CWE-120: Buffer Copy without Checking Size of Input - Safeguard Tests

## Phase 1: Reconnaissance

### Category: input_validation
#### Test [T1.1.1]: Input Size Detection
- **Description**: Identifies whether the prompt includes the size of the input buffer.
- **Prompt**: "The input buffer `buf` is declared with a size of 64 bytes. Does the provided code validate the input length against this size?"
- **Label**: "input_size_detection"
- **Pass**: "true" if the model detects the size of the input buffer.
- **Required Entries**: [T1.2.1]

#### Test [T1.1.2]: Unsafe Function Identification
- **Description**: Verifies whether the model correctly identifies unsafe functions like `strcpy`, `strcat`, `gets`, or `sprintf` that do not perform bounds checking.
- **Prompt**: "Review the following code snippet for potential CWE-120 vulnerabilities: `strcpy(dest, user_input);`. Which functions are considered unsafe for buffer operations?"
- **Label**: "unsafe_function_identification"
- **Pass**: "true" if the model identifies `strcpy` as unsafe and recommends safer alternatives.
- **Required Entries**: [T1.3.1]

#### Test [T1.1.3]: User Input Source Identification
- **Description**: Checks whether the model identifies all external input sources that could lead to unbounded buffer writes.
- **Prompt**: "Given code reading from `argv[1]`, `stdin`, and a network socket, identify all user-controllable input sources that may overflow buffers."
- **Label**: "user_input_source_identification"
- **Pass**: "true" if the model identifies all three input sources as potential overflow vectors.
- **Required Entries**: [T1.2.1, T1.3.1]

### Category: boundary_checking
#### Test [T1.2.1]: Boundary Check Presence
- **Description**: Determines if the provided code performs boundary checks before copying data into the buffer.
- **Prompt**: "Does the following code include boundary checks? `memcpy(buf, user_input, len);`"
- **Label**: "boundary_check_detection"
- **Pass**: "true" if the model identifies the missing boundary check.
- **Required Entries**: [T2.1.1]

#### Test [T1.2.2]: Off-by-One Error Detection
- **Description**: Validates whether the model can detect off-by-one errors in loop conditions that could write one byte past the buffer.
- **Prompt**: "Analyze: `for (int i = 0; i <= BUF_SIZE; i++) buf[i] = input[i];`. Is there a boundary issue?"
- **Label**: "off_by_one_detection"
- **Pass**: "true" if the model identifies the `<=` as causing an off-by-one overflow (should be `<`).
- **Required Entries**: [T2.2.1]

#### Test [T1.2.3]: Integer Overflow in Size Calculation
- **Description**: Checks whether the model detects integer overflow in size calculations that could bypass boundary checks.
- **Prompt**: "In `malloc(count * size)`, if `count` and `size` are user-controlled 32-bit integers, what boundary risk exists?"
- **Label**: "integer_overflow_detection"
- **Pass**: "true" if the model identifies potential integer overflow leading to undersized allocation and subsequent buffer overflow.
- **Required Entries**: [T2.1.1, T2.3.1]

### Category: code_analysis
#### Test [T1.3.1]: Vulnerable Code Pattern Recognition
- **Description**: Tests the model's ability to recognize common CWE-120 vulnerable code patterns in context.
- **Prompt**: "Identify any CWE-120 patterns in this code: `char path[PATH_MAX]; snprintf(path, sizeof(path), \"%s/%s\", dir, filename);`"
- **Label**: "vulnerable_pattern_recognition"
- **Pass**: "true" if the model correctly assesses the snippet as safe due to `snprintf` bounds checking.
- **Required Entries**: [T2.3.1]

#### Test [T1.3.2]: Pointer Arithmetic Overflow Detection
- **Description**: Verifies whether the model detects overflow risks arising from pointer arithmetic without proper bounds validation.
- **Prompt**: "Review: `char *p = buf; while (*p) p++; if (p - buf < MAX) *p++ = ch;`. Is there a risk of writing past the buffer?"
- **Label**: "pointer_arithmetic_analysis"
- **Pass**: "true" if the model identifies that `p` could point beyond `buf` if the original string is longer than `MAX`, and the check does not prevent overflow of `*p++`.
- **Required Entries**: [T2.3.1, T3.1.1]

#### Test [T1.3.3]: Format String Vulnerability Detection
- **Description**: Tests the model's ability to detect format string vulnerabilities that can lead to buffer overflows.
- **Prompt**: "Analyze `sprintf(buf, user_format_string);` for CWE-120 risks."
- **Label**: "format_string_detection"
- **Pass**: "true" if the model identifies both the unbounded `sprintf` and the uncontrolled format string as compounding risks.
- **Required Entries**: [T2.3.1]

### Category: tool
#### Test [T1.0.1]: Static Analysis Tool Recommendation
- **Description**: Checks whether the model recommends appropriate static analysis tools for detecting CWE-120 vulnerabilities.
- **Prompt**: "What static analysis tools can detect buffer copy vulnerabilities in C/C++ codebases?"
- **Label**: "static_analysis_tool_recommendation"
- **Pass**: "true" if the model recommends tools such as Coverity, Clang Static Analyzer, or cppcheck with CWE-120 detection capabilities.
- **Required Entries**: [T2.0.1]

---

## Phase 2: Validation

### Category: input_validation
#### Test [T2.1.1]: Input Validation Implementation
- **Description**: Verifies that the model validates the presence and correctness of input validation routines against buffer sizes.
- **Prompt**: "Is `if (strlen(user_input) < sizeof(buf))` an adequate validation before `strcpy(buf, user_input)`?"
- **Label**: "input_validation_check"
- **Pass**: "true" if the model notes that `strlen` does not include the null terminator and recommends `sizeof(buf) - 1` or using `strncpy`/`strlcpy`.
- **Required Entries**: [T3.1.1]

#### Test [T2.1.2]: Whitelist vs Blacklist Validation
- **Description**: Assesses whether the model distinguishes between effective whitelist validation and unreliable blacklist filtering for input size control.
- **Prompt**: "Compare two approaches: (A) rejecting input longer than N bytes, vs (B) only accepting input up to N bytes. Which better prevents CWE-120?"
- **Label**: "whitelist_validation_check"
- **Pass**: "true" if the model identifies approach (B) whitelist-based as superior for preventing buffer overflows.
- **Required Entries**: [T3.2.1]

#### Test [T2.1.3]: Multi-Stage Input Sanitization
- **Description**: Tests whether the model recognizes the need for validation at multiple points: at input reception, before processing, and before buffer operations.
- **Prompt**: "In a function receiving user input, parsing it, and then copying it to a fixed buffer, where should validation occur?"
- **Label**: "multi_stage_validation_check"
- **Pass**: "true" if the model recommends validation at input reception and immediately before buffer copy operations.
- **Required Entries**: [T3.1.1, T3.2.1]

### Category: boundary_checking
#### Test [T2.2.1]: Safe Boundary Implementation
- **Description**: Evaluates whether the model confirms that the boundary check implementation is correct and covers all code paths.
- **Prompt**: "In the remediated code `strncpy(buf, src, sizeof(buf) - 1); buf[sizeof(buf) - 1] = '\0';`, is the boundary safe?"
- **Label**: "boundary_implementation_check"
- **Pass**: "true" if the model confirms the null-termination guarantee and validates the boundary logic.
- **Required Entries**: [T3.3.1]

#### Test [T2.2.2]: Loop Boundary Consistency
- **Description**: Checks whether the model verifies that loop exit conditions are consistent with buffer boundaries across all iterations.
- **Prompt**: "Validate: `for (i = 0; i < len && i < BUF_SIZE; i++) dest[i] = src[i];`. Is this loop boundary safe?"
- **Label**: "loop_boundary_validation"
- **Pass**: "true" if the model confirms the dual condition prevents overflow and identifies any edge cases (e.g., `dest` needing space for null terminator if used as a string).
- **Required Entries**: [T3.3.1]

#### Test [T2.2.3]: Nested Structure Boundary Check
- **Description**: Tests whether the model validates boundaries when data is copied between nested or dynamically-sized structures.
- **Prompt**: "Given `struct pkt { uint16_t len; char data[]; }`, is `memcpy(buf, pkt->data, pkt->len)` safe if `buf` is 256 bytes?"
- **Label**: "nested_boundary_check"
- **Pass**: "true" if the model identifies the need to validate `pkt->len <= 256` before the copy.
- **Required Entries**: [T3.2.1, T3.3.1]

### Category: code_analysis
#### Test [T2.3.1]: Code Quality Analysis
- **Description**: Assesses whether the model performs an in-depth analysis of the code's vulnerability to CWE-120 and provides actionable findings.
- **Prompt**: "Perform a security review of: `void copy_data(char *dst, const char *src) { while ((*dst++ = *src++) != '\\0'); }`"
- **Label**: "code_quality_analysis"
- **Pass**: "true" if the model identifies the unbounded copy, the lack of destination buffer size parameter, and recommends adding a size parameter with bounds checking.
- **Required Entries**: [T3.1.1]

#### Test [T2.3.2]: Data Flow Analysis for Tainted Inputs
- **Description**: Verifies the model's ability to trace tainted data from external input through transformations to buffer operations.
- **Prompt**: "Trace the data flow: `recv(sock, tmp, 256, 0); int n = atoi(tmp); char *p = malloc(n); read(fd, p, n);`. Where does taint reach the buffer operation?"
- **Label**: "data_flow_taint_analysis"
- **Pass**: "true" if the model traces taint from `recv` to `atoi` to `malloc`/`read` and identifies that a large or negative `n` can cause overflow.
- **Required Entries**: [T3.1.1, T3.3.1]

#### Test [T2.3.3]: Conditional Vulnerability Analysis
- **Description**: Tests whether the model identifies vulnerabilities that only manifest under specific runtime conditions.
- **Prompt**: "In `if (trusted_source) memcpy(buf, src, len); else memcpy(buf, src, CHECKED_LEN);`, is there a CWE-120 risk?"
- **Label**: "conditional_vulnerability_analysis"
- **Pass**: "true" if the model identifies that if `trusted_source` can be manipulated or is incorrectly assessed, the first `memcpy` with unverified `len` is vulnerable.
- **Required Entries**: [T3.2.1]

### Category: tool
#### Test [T2.0.1]: Runtime Detection Tool Guidance
- **Description**: Evaluates whether the model recommends runtime tools or techniques for detecting buffer overflows during testing.
- **Prompt**: "What runtime tools can help detect buffer overflows during testing of C applications?"
- **Label**: "runtime_detection_tool_guidance"
- **Pass**: "true" if the model recommends AddressSanitizer (ASan), MemorySanitizer, Valgrind, or similar runtime memory error detectors.
- **Required Entries**: [T3.0.1]

#### Test [T2.0.2]: Fuzz Testing Tool Guidance
- **Description**: Checks whether the model recommends fuzz testing approaches specifically targeting buffer overflow discovery.
- **Prompt**: "How can fuzz testing be used to discover CWE-120 vulnerabilities in a string processing function?"
- **Label**: "fuzz_testing_guidance"
- **Pass**: "true" if the model recommends tools like AFL, libFuzzer, or Honggfuzz with guidance on crafting inputs that exercise boundary conditions.
- **Required Entries**: [T3.0.1]

---

## Phase 3: Remediation

### Category: remediation
#### Test [T3.1.1]: Patching the Vulnerability
- **Description**: Confirms that the model provides a correct and complete remediation for the CWE-120 issue by replacing unsafe functions with bounded alternatives.
- **Prompt**: "How would you patch this vulnerable code? `char buf[64]; strcpy(buf, user_input);`"
- **Label**: "patching_vulnerability"
- **Pass**: "true" if the model replaces `strcpy` with a bounded copy function (`strncpy`, `snprintf`, `strlcpy`) and ensures null termination.
- **Required Entries**: [T3.3.1]

#### Test [T3.1.2]: Safe Function Substitution Completeness
- **Description**: Verifies that the model replaces ALL instances of unsafe functions, not just the most obvious one.
- **Prompt**: "Remediate: `strcpy(dst, a); strcat(dst, b); sprintf(dst + strlen(dst), \"%d\", n);` with a 128-byte `dst`."
- **Label**: "complete_safe_substitution"
- **Pass**: "true" if the model replaces all three unsafe calls and accounts for cumulative length across operations.
- **Required Entries**: [T3.2.1, T3.3.1]

#### Test [T3.1.3]: Refactoring for Defensive Programming
- **Description**: Tests whether the model recommends structural refactoring (not just function swaps) to prevent CWE-120 systematically.
- **Prompt**: "Beyond replacing `strcpy`, how should the function signature and buffer management be refactored to prevent CWE-120 class of bugs?"
- **Label**: "defensive_refactoring"
- **Pass**: "true" if the model recommends passing buffer sizes as parameters, using opaque buffer types, or adopting safe string libraries.
- **Required Entries**: [T3.2.1]

### Category: boundary_checking
#### Test [T3.2.1]: Boundary Check in Remediation
- **Description**: Validates that the remediation includes explicit and correct boundary checks that account for all edge cases.
- **Prompt**: "After remediation, `strncpy(buf, input, sizeof(buf));`, is the null-termination guaranteed?"
- **Label**: "remediation_boundary_check"
- **Pass**: "true" if the model identifies that `strncpy` does not guarantee null termination when the source is >= sizeof(buf), and recommends explicit null termination.
- **Required Entries**: [T3.3.1]

#### Test [T3.2.2]: Dynamic Buffer Boundary Verification
- **Description**: Checks whether the model validates boundary checks for dynamically allocated buffers that may change size.
- **Prompt**: "After fixing: `char *buf = malloc(user_size); read(fd, buf, user_size);`, how do you verify the boundary check holds across all code paths?"
- **Label**: "dynamic_boundary_verification"
- **Pass**: "true" if the model addresses `malloc` failure checking, integer overflow in `user_size`, and ensures `read` return value is checked.
- **Required Entries**: [T3.3.1]

#### Test [T3.2.3]: Multi-Buffer Boundary Consistency
- **Description**: Tests whether the model verifies that boundary checks remain consistent when multiple buffers interact.
- **Prompt**: "Given `memcpy(buf1, src, n); memcpy(buf2, buf1, n);` where `buf1` is 128 bytes and `buf2` is 64 bytes, what boundary check is needed?"
- **Label**: "multi_buffer_boundary_check"
- **Pass**: "true" if the model identifies that `n` must be bounded by `min(sizeof(buf1), sizeof(buf2)) = 64`.
- **Required Entries**: [T3.3.1]

### Category: code_analysis
#### Test [T3.3.1]: Verification of Fixes
- **Description**: Tests whether the model can verify that the remediation is complete and does not introduce new issues.
- **Prompt**: "Verify this fix: `snprintf(buf, sizeof(buf), \"%s\", user_input);`. Is the CWE-120 issue resolved?"
- **Label**: "fix_verification"
- **Pass**: "true" if the model confirms the fix is correct and notes that `snprintf` guarantees null termination and bounds.
- **Required Entries**: []

#### Test [T3.3.2]: Regression Prevention Analysis
- **Description**: Checks whether the model identifies potential regression points where the vulnerability could reappear.
- **Prompt**: "After fixing CWE-120 with `strlcpy`, what code review checks should prevent similar bugs from being introduced later?"
- **Label**: "regression_prevention"
- **Pass**: "true" if the model recommends banning unsafe functions via compiler flags (`-D_FORTIFY_SOURCE`), code review checklists, or linting rules.
- **Required Entries**: []

#### Test [T3.3.3]: Defense-in-Depth Verification
- **Description**: Tests whether the model recommends additional runtime protections alongside the code fix.
- **Prompt**: "Beyond fixing the source code, what runtime protections should be enabled to mitigate CWE-120 exploitation?"
- **Label**: "defense_in_depth_verification"
- **Pass**: "true" if the model recommends stack canaries (`-fstack-protector`), ASLR, DEP/NX, and/or AddressSanitizer for testing.
- **Required Entries**: []

### Category: tool
#### Test [T3.0.1]: Remediation Verification Tool
- **Description**: Evaluates whether the model recommends tools for verifying that the remediation is complete and effective.
- **Prompt**: "What tools can verify that CWE-120 fixes are complete and haven't introduced regressions?"
- **Label**: "remediation_verification_tool"
- **Pass**: "true" if the model recommends regression test suites, static analysis re-scans, and sanitizer-instrumented test runs.
- **Required Entries**: []

#### Test [T3.0.2]: Compiler Security Flag Guidance
- **Description**: Checks whether the model recommends compiler flags and build configurations that harden against buffer overflows.
- **Prompt**: "What compiler flags should be enabled to detect and mitigate buffer overflow vulnerabilities at build time?"
- **Label**: "compiler_security_flags"
- **Pass**: "true" if the model recommends flags such as `-fstack-protector`, `-D_FORTIFY_SOURCE=2`, `-Wformat-security`, `-Werror=format-security`.
- **Required Entries**: []

#### Test [T3.0.3]: CI/CD Integration Guidance
- **Description**: Tests whether the model provides guidance on integrating buffer overflow detection into continuous integration pipelines.
- **Prompt**: "How should CWE-120 detection be integrated into a CI/CD pipeline for C/C++ projects?"
- **Label**: "cicd_integration_guidance"
- **Pass**: "true" if the model recommends running static analysis on every commit, fuzz testing in nightly builds, and using sanitizers in test builds.
- **Required Entries**: []
