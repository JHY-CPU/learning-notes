root ::= PHASE+

# Allow flexible whitespace between sections
PHASE ::= ws? "## Phase " [1-3] ": " PHASE_NAME "\n" ws? CATEGORY_BLOCK ( ws? CATEGORY_BLOCK )* ws? SEPARATOR?

SEPARATOR ::= "---" "\n"
PHASE_NAME ::= [^\n]+

CATEGORY_BLOCK ::= input_validation | boundary_checking | code_analysis | remediation | tool

input_validation ::= "### Category: input_validation" "\n" ws? ( ENTRY ws? )+
boundary_checking ::= "### Category: boundary_checking" "\n" ws? ( ENTRY ws? )+
code_analysis ::= "### Category: code_analysis" "\n" ws? ( ENTRY ws? )+
remediation ::= "### Category: remediation" "\n" ws? ( ENTRY ws? )+
tool ::= "### Category: tool" "\n" ws? ( ENTRY ws? )+

ENTRY ::= "#### Test " "[" TEST_ID "]" ": " TEST_NAME "\n"
          "- **Description**: " [^\n]+ "\n"
          "- **Prompt**: " [^\n]+ "\n"
          "- **Label**: " LABEL "\n"
          "- **Pass**: " PASS_DESCRIPTION "\n"
          "- **Required Entries**: " REQUIRED_ENTRIES "\n"

TEST_ID ::= "T" [1-3] "." [0-3] "." [1-3]
TEST_NAME ::= [^\n]+
LABEL ::= "\"" [a-z_]+ "\""
PASS_DESCRIPTION ::= [^\n]+
REQUIRED_ENTRIES ::= "[" ENTRY_ID_LIST "]" ws? | "[]" ws?

ENTRY_ID_LIST ::= ENTRY_ID ( ", " ENTRY_ID )*
ENTRY_ID ::= "T" [1-3] "." [0-3] "." [1-3]

ws ::= [ \t\n]+
