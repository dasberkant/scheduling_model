import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import io  # for handling in-memory plot objects

def solve_rental_scheduling_streamlit(num_orders, num_vehicles,
                                      p_list, d_list, e_list, l_list,
                                      A_list, big_m=100):
    """
    Solves the one-day rental scheduling model with:
      - num_orders: how many orders (I=1..n)
      - num_vehicles: how many vehicles (V=1..m)
      - p_list[i]: processing time for order i
      - d_list[i]: deadline for order i
      - e_list[i]: earliest start (time window)
      - l_list[i]: latest finish (time window)
      - A_list[v]: availability time for vehicle v
      - big_m: a large constant for the model

    Returns a dict with solution info: x, z, S, leftover, objective, gantt data.
    Or raises an exception if no feasible solution is found.
    """

    # Define sets
    I = list(range(1, num_orders+1))
    V = list(range(1, num_vehicles+1))

    # Convert the user inputs into a dict A[v]
    A = {v: A_list[v-1] for v in V}

    # Create a Gurobi model
    model = gp.Model("RentalSchedulingStreamlit")
    model.setParam("OutputFlag", 0)  # turn off solver printing in Streamlit UI

    # Decision variables
    x = model.addVars(I, V, vtype=GRB.BINARY, name="x")
    z = model.addVars(I, vtype=GRB.BINARY, name="z")
    S = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="S")
    Lvar = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="L")  # leftover margin
    delta = model.addVars(I, I, V, vtype=GRB.BINARY, name="delta")

    # 1) Assign or Backorder
    for i in I:
        model.addConstr(gp.quicksum(x[i,v] for v in V) + z[i] == 1,
                        name=f"AssignOrBack_{i}")

    # 2) Mandatory Delivery if 24 + p[i] > d[i]
    #    => must do today (z[i]=0), finish by d[i].
    for i in I:
        if 24 + p_list[i-1] > d_list[i-1]:
            model.addConstr(z[i] == 0, name=f"MustToday_z_{i}")
            model.addConstr(S[i] + p_list[i-1] <= d_list[i-1], name=f"MustFinish_{i}")

    # 3) Vehicle availability
    #    S[i] >= A[v] if x[i,v]=1 (via big-M form).
    for i in I:
        for v in V:
            model.addConstr(
                S[i] >= A[v] - big_m*(1 - x[i,v]),
                name=f"Avail_{i}_{v}"
            )

    # 4) No Overlap constraints
    for i in I:
        for j in I:
            if i < j:
                for v in V:
                    model.addConstr(delta[i,j,v] <= x[i,v] + x[j,v],
                                    name=f"deltaActive_{i}_{j}_{v}")
                    model.addConstr(
                        S[j] >= S[i] + p_list[i-1] - big_m*(1 - delta[i,j,v]),
                        name=f"NoOverlap1_{i}_{j}_{v}"
                    )
                    model.addConstr(
                        S[i] >= S[j] + p_list[j-1] - big_m*delta[i,j,v],
                        name=f"NoOverlap2_{i}_{j}_{v}"
                    )

    # 5) 24-Hour Start Limit if 24 + p[i] <= d[i]
    for i in I:
        if 24 + p_list[i-1] <= d_list[i-1]:
            model.addConstr(
                S[i] <= 24 + big_m*z[i],
                name=f"StartBefore24_{i}"
            )

    # 6) Leftover margin definition
    for i in I:
        model.addConstr(
            Lvar[i] >= d_list[i-1] - (S[i] + p_list[i-1]) - big_m*z[i],
            name=f"Lmin_{i}"
        )
        model.addConstr(
            Lvar[i] <= d_list[i-1] - (S[i] + p_list[i-1]) + big_m*z[i],
            name=f"Lmax_{i}"
        )
        model.addConstr(
            Lvar[i] <= big_m*(1 - z[i]),
            name=f"LnonNeg_{i}"
        )

    # 7) Time Window constraints
    for i in I:
        model.addConstr(
            S[i] >= e_list[i-1] - big_m*z[i],
            name=f"TimeWinStart_{i}"
        )
        model.addConstr(
            S[i] + p_list[i-1] <= l_list[i-1] + big_m*z[i],
            name=f"TimeWinFinish_{i}"
        )

    # Objective: Max leftover margin
    model.setObjective(gp.quicksum(Lvar[i] for i in I), GRB.MAXIMIZE)

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # Build results
        x_sol = {}
        z_sol = {}
        S_sol = {}
        L_sol = {}
        for i in I:
            z_sol[i] = int(z[i].X + 0.5)  # 0 or 1
            S_sol[i] = S[i].X
            L_sol[i] = Lvar[i].X
            for v in V:
                if x[i,v].X > 0.5:
                    x_sol[i] = v

        # Gantt data
        gantt_data = {}
        for v in V:
            gantt_data[v] = []
            for i in I:
                if i in x_sol and x_sol[i] == v:
                    gantt_data[v].append((i, S_sol[i], p_list[i-1]))

        return {
            "objective": model.ObjVal,
            "x": x_sol,
            "z": z_sol,
            "S": S_sol,
            "L": L_sol,
            "gantt_data": gantt_data
        }
    else:
        raise ValueError("No feasible or optimal solution found.")

def main():
    st.title("One-Day Rental Scheduling with Time Windows")
    st.write("""
    This app allows you to input order data for a single-day scheduling model, 
    specify vehicles' availability times, then uses Gurobi to solve and display results.
    """)

    # === Sidebar: Model Settings ===
    st.sidebar.header("Model Settings")
    num_orders = st.sidebar.number_input("Number of Orders", min_value=1, max_value=20, value=5)
    num_vehicles = st.sidebar.number_input("Number of Vehicles", min_value=1, max_value=10, value=2)

    st.sidebar.write("You may change the Big-M if desired:")
    big_m = st.sidebar.number_input("Big-M value", value=100, step=50)

    st.markdown("---")

    # === Vehicle Availability Input ===
    st.subheader("Vehicle Availability")
    st.write("For each vehicle, enter the hour at which it becomes available (A[v]).")
    A_list = []
    for v in range(num_vehicles):
        val = st.number_input(f"Availability of Vehicle {v+1}", min_value=0.0, value=0.0, step=1.0)
        A_list.append(val)

    st.markdown("---")

    # === Order Data Input ===
    st.subheader("Order Data")
    st.write("Enter each order's Processing Time, Deadline, Earliest Start, Latest Finish.")
    
    p_list = []
    d_list = []
    e_list = []
    l_list = []

    for i in range(num_orders):
        cols = st.columns(4)
        with cols[0]:
            p_val = st.number_input(f"p[{i+1}]:", min_value=0.0, value=2.0, step=1.0)
        with cols[1]:
            d_val = st.number_input(f"d[{i+1}]:", min_value=0.0, value=24.0, step=1.0)
        with cols[2]:
            e_val = st.number_input(f"E[{i+1}]:", min_value=0.0, value=0.0, step=1.0)
        with cols[3]:
            l_val = st.number_input(f"L[{i+1}]:", min_value=0.0, value=999.0, step=1.0)

        p_list.append(p_val)
        d_list.append(d_val)
        e_list.append(e_val)
        l_list.append(l_val)

    # === Solve Button ===
    if st.button("Solve"):
        try:
            result = solve_rental_scheduling_streamlit(
                num_orders, num_vehicles,
                p_list, d_list, e_list, l_list,
                A_list,
                big_m=big_m
            )

            st.success("Solution found!")
            st.write(f"**Objective (Total Leftover Margin)** = {result['objective']:.2f}")

            # Detailed Results
            st.write("### Detailed Results:")
            for i in range(1, num_orders+1):
                z_i = result["z"][i]
                leftover = result["L"][i]
                if i in result["x"]:
                    assigned_vehicle = result["x"][i]
                    start_time = result["S"][i]
                    st.write(f"- **Order {i}**: done today (z=0), vehicle={assigned_vehicle}, "
                             f"start={start_time:.2f}, leftover={leftover:.2f}")
                else:
                    st.write(f"- **Order {i}**: backordered (z=1), leftover=0")

            # Gantt Charts per vehicle
            # -- Inside your loop over vehicles:
            for v in range(1, num_vehicles+1):
                if len(result["gantt_data"][v]) == 0:
                    st.write(f"**Vehicle {v}:** No assigned orders.")
                    continue

                st.write(f"## Gantt Chart: Vehicle {v}")

                # Sort tasks by start time
                tasks = sorted(result["gantt_data"][v], key=lambda t: t[1])

                fig, ax = plt.subplots()

                # We'll track the maximum finishing time among all tasks
                max_end_time = 24

                y_ticks_labels = []
                for idx, (order_id, stime, duration) in enumerate(tasks):
                    # Plot horizontal bar
                    ax.barh(idx, duration, left=stime)

                    # Build a label
                    label_text = f"Order {order_id}\nStart={stime:.1f}, Dur={duration}"
                    
                    # Decide whether text fits inside or outside
                    threshold = 1.5
                    if duration < threshold:
                        text_x = stime + duration + 0.1
                        text_ha = 'left'
                        text_color = 'black'
                    else:
                        text_x = stime + duration/2
                        text_ha = 'center'
                        text_color = 'white'
                    ax.text(text_x, idx, label_text, ha=text_ha, va='center',
                            color=text_color, fontsize=8)

                    # Track the end time to set chart scale
                    end_time = stime + duration
                    if end_time > max_end_time:
                        max_end_time = end_time

                    y_ticks_labels.append(f"Order {order_id}")

                # Now set the x-axis from 0 to the maximum finishing time (or 24 if bigger)
                ax.set_xlim(0, max_end_time)

                ax.set_xlabel("Time (hours)")
                ax.set_ylabel("Orders")
                ax.set_title(f"Vehicle {v} Gantt")
                ax.set_yticks(range(len(tasks)))
                ax.set_yticklabels(y_ticks_labels)

                st.pyplot(fig)


        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
