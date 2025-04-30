-- TPC-H Query 10

select
        c_custkey,
        sum(l_extendedprice * (1 - l_discount)) as revenue,
        c_acctbal
from
        customer,
        orders,
        lineitem,
        nation
where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate >= date '1993-10-01'
        and o_orderdate < date '1994-01-01'
        and l_returnflag = 'R'
        and c_nationkey = n_nationkey
group by
        c_custkey,
        c_acctbal
order by
        revenue desc
