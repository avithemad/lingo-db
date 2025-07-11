-- TPC-H Query 13

select
        c_count,
        count(*) as custdist
from
        (
                select
                        c_custkey,
                        count(o_orderkey) c_count
                from
                        customer join orders on
                                c_custkey = o_custkey
                group by
                        c_custkey
        ) as c_orders
group by
        c_count
order by
        custdist desc,
        c_count desc
