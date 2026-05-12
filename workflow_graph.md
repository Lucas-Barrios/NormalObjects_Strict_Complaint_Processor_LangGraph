# NormalObjects — Bloyce's Protocol Workflow

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	intake(intake)
	validate(validate)
	investigate(investigate)
	resolve(resolve)
	close(close)
	__end__([<p>__end__</p>]):::last
	__start__ --> intake;
	intake -.-> __end__;
	intake -.-> validate;
	investigate -.-> __end__;
	investigate -.-> resolve;
	resolve -.-> __end__;
	resolve -.-> close;
	validate -.-> __end__;
	validate -.-> investigate;
	close --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
