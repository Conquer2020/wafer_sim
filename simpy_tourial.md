#### Simpy 事件注册机制

```
strong
```

Result:

process 1 done @20.000

process 2 done @30.000

process overlap_process done @30.000

process 1 done @40.000

process short_process done @50.000

process 2 done @70.000

process order_process done @70.000
