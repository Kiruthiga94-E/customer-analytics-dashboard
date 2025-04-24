# 1. Import Libraries

import pandas as pd

# Dashboard Dependencies
import dash
from dash import html, dcc
from dash.dependencies import Input, Output

# Plotly Dependencies
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

import re
pio.templates.default = "plotly_dark"


# 2. Load Data
df = pd.read_csv("ecommerce_customer_data.csv")  # Replace with your actual data file
frequent_itemsets = pd.read_csv('frequent_itemsets.csv')
rules = pd.read_csv('association_rules.csv')


# 3. Initialize App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "E-commerce Customer Analytics Dashboard"

# 4. Layout with Tabs
app.layout = html.Div([
    html.H1("Customer Behaviour Analytics Dashboard", style={
        'textAlign': 'center',
        'marginBottom': '30px',
        'marginTop': '30px',
        'color': '#ecf0f1'
    }),

    dcc.Tabs(id="tabs", value='tab1', children=[
        dcc.Tab(label='Customer Insights', value='tab1'),
        dcc.Tab(label='Sales Insights', value='tab2'),
        dcc.Tab(label='Customer Behavior', value='tab3'),
        dcc.Tab(label='Product Performance', value='tab4'),
        dcc.Tab(label='Market Basket Analysis', value='tab5')
    ],
    colors={
            "border": "#2c3e50",
            "primary": "#2980b9",
            "background": "#6200ee"
        },
        style={'color': '#ecf0f1', 'fontSize': '18px'}
    ), 
    html.Div(id='tabs-content')
], style={
    'backgroundColor': '#1e272e',
    'minHeight': '100vh',
    'padding': '20px',
    'fontFamily': 'Poppins, sans-serif'
})

# 5. Callbacks for Tabs
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab1':
        # Age Distribution Histogram
        age_fig = px.histogram(df, x="Customer Age", nbins=20, color_discrete_sequence=["#6A5ACD"],
                               title="Customer Age Distribution", text_auto=True)
        age_fig.update_layout(bargap=0.1)

        # Gender Distribution Pie Chart
        gender_counts = df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        gender_fig = px.pie(gender_counts, names='Gender', values='Count', title='Gender Distribution',
                            color_discrete_sequence=px.colors.sequential.RdBu)

        # Total Purchase by Gender Bar Chart
        gender_purchase = df.groupby('Gender')['Total Purchase Amount'].sum().reset_index()
        gender_purchase_fig = px.bar(gender_purchase, x='Gender', y='Total Purchase Amount',
                                     title='Total Purchase Amount by Gender', text='Total Purchase Amount',
                                     color='Gender', color_discrete_sequence=['#FF8C00', '#6A5ACD'])
        gender_purchase_fig.update_layout(bargap=0.5),
        # CLV Segment Pie Chart
        clv_seg = df['CLV Segment'].value_counts().reset_index()
        clv_seg.columns = ['CLV Segment', 'Count']
        clv_fig = px.pie(clv_seg, names='CLV Segment', values='Count', title='Customer Segments by CLV',
                         color_discrete_sequence=['#FF8C00', '#6A5ACD', '#00A572'])

        return html.Div([
                html.Div([
                    html.H4("Customer Insights Recommendations", style={'marginBottom': '15px', 'color': '#fbc531'}),
                    html.Div(id='customer-behaviour-recommendations-card')
                ], style={
                    'marginTop': '30px',
                    'padding': '20px',
                    'backgroundColor': '#2f3640',
                    'color': '#f5f6fa',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.5)',
                    'fontSize': '18px'
                }),
            html.Div([
                html.Div([dcc.Graph(figure=clv_fig)], style={'width': '48%', 'padding': '10px'}),
                html.Div([dcc.Graph(figure=gender_fig)], style={'width': '48%', 'padding': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),

            html.Div([
                html.Div([dcc.Graph(figure=age_fig)], style={'width': '48%', 'padding': '10px'}),
                html.Div([dcc.Graph(figure=gender_purchase_fig)], style={'width': '48%', 'padding': '10px'}),
                
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ])
    
    elif tab == 'tab2':
        return tab2_layout
    elif tab == 'tab3':
        return tab3_layout
    
    elif tab == 'tab4':
        return tab4_layout
    
    elif tab == 'tab5':
        return tab5_layout

# Accurate CLV calculation
df['Total Purchase Amount'] = df['Total Purchase Amount'].astype(float)
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])

clv_df = df.groupby('Customer ID').agg({
    'Total Purchase Amount': 'sum',
    'Purchase Date': lambda x: (x.max() - x.min()).days + 1,
    'Customer ID': 'count'
}).rename(columns={
    'Total Purchase Amount': 'Revenue',
    'Purchase Date': 'Lifespan',
    'Customer ID': 'Frequency'
})

clv_df['CLV'] = clv_df['Revenue'] * (clv_df['Frequency'] / clv_df['Lifespan'])

# Merge CLV values into main df
df = df.merge(clv_df['CLV'], on='Customer ID', how='left')

# Segment based on CLV
df['CLV Segment'] = pd.qcut(df['CLV'], q=3, labels=['Low Value', 'Mid Value', 'High Value'])

@app.callback(
    Output('customer-behaviour-recommendations-card', 'children'),
    Input('tabs', 'value')
)
def update_customer_insight_recommendations(tab):
    if tab != 'tab1':
        raise dash.exceptions.PreventUpdate

    recommendations = []

    # Age distribution insight
    age_counts = df['Customer Age'].value_counts().sort_index()
    if any(age_counts[age] < 0.05 * age_counts.sum() for age in range(18, 20)):
        recommendations.append("📉 Consider targeted campaigns to attract younger customers (below 20), who currently form a smaller part of your user base.")

    # CLV segmentation logic
        clv_counts = df['CLV Segment'].value_counts(normalize=True)

        low_pct = clv_counts.get('Low', 0)
        mid_pct = clv_counts.get('Mid', 0)
        high_pct = clv_counts.get('High', 0)

        # Check for near-even distribution (e.g., all around 33%)
        if abs(low_pct - mid_pct) < 0.07 and abs(mid_pct - high_pct) < 0.07 and abs(low_pct - high_pct) < 0.07:
            recommendations.append("🔍 CLV segments are nearly evenly distributed. Consider loyalty initiatives or personalized campaigns to encourage more customers into the High CLV category.")
        elif high_pct > 0.40:
            recommendations.append("🚀 A large proportion of your customers are High CLV. Focus on retaining these valuable customers with exclusive benefits.")
        elif low_pct > 0.40:
            recommendations.append("⚠️ Majority of your customers are in the Low CLV segment. Introduce engagement strategies to increase lifetime value.")

    if not recommendations:
        recommendations.append("✅ Customer segments are well balanced. Keep monitoring trends for shifts in behavior.")

    return html.Ul([html.Li(rec, style={'marginBottom': '10px'}) for rec in recommendations])



# Tab 2: Sales Insights Layout
tab2_layout = html.Div([
    html.Div([
    html.H4("Sales Recommendations", style={'marginBottom': '15px', 'color': '#fbc531'}),
        html.Div(id='sales-recommendation-box')
    ], style={
        'marginTop': '30px',
        'marginBottom': '30px',
        'padding': '20px',
        'backgroundColor': '#2f3640',
        'color': '#f5f6fa',
        'borderRadius': '10px',
        'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.5)',
        'fontSize': '18px'
    }),
    # Date Range Picker
    html.Div([
        html.Label("Select Date Range:", style={'color': 'white', 'marginRight' : '20px', 'fontSize': '18px'}),
        dcc.DatePickerRange(
            id='sales-date-range',
            min_date_allowed=df['Purchase Date'].min(),
            max_date_allowed=df['Purchase Date'].max(),
            start_date=df['Purchase Date'].min(),
            end_date=df['Purchase Date'].max()
        )
    ], style={'margin': '10px'}),

    # Sales Revenue Over Time
    dcc.Graph(id='sales-time-series'),

    html.Div([
        dcc.Graph(id='age-trend-line', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='gender-trend-line', style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),

    html.Div([
        dcc.Graph(id='category-sales-bar', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='gender-sales-bar', style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'})
])

@app.callback(
    [Output('sales-time-series', 'figure'),
     Output('category-sales-bar', 'figure'),
     Output('gender-sales-bar', 'figure'),
     Output('age-trend-line', 'figure'),
     Output('gender-trend-line', 'figure')],
    [Input('sales-date-range', 'start_date'),
     Input('sales-date-range', 'end_date')]
)
def update_sales_insights(start_date, end_date):
    filtered_df = df[
        (df['Purchase Date'] >= pd.to_datetime(start_date)) &
        (df['Purchase Date'] <= pd.to_datetime(end_date))
    ]

    # Line Chart: Sales Over Time
    ts_data = filtered_df.groupby(filtered_df['Purchase Date'].dt.to_period('M'))['Total Purchase Amount'].sum().reset_index()
    ts_data['Purchase Date'] = ts_data['Purchase Date'].astype(str)
    ts_fig = px.line(ts_data, x='Purchase Date', y='Total Purchase Amount',
                     title="Total Sales Over Time", markers=True)

    # Bar Chart: Sales by Category
    cat_data = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
    cat_fig = px.bar(cat_data, x='Product Category', y='Total Purchase Amount',
                     title="Sales by Product Category", color='Product Category')
    
    # 3. Bar Chart: Sales by Gender
    gender_data = filtered_df.groupby('Gender')['Total Purchase Amount'].sum().reset_index()
    gender_fig = px.bar(gender_data, x='Gender', y='Total Purchase Amount',
                        title="Sales by Gender", color='Gender')
    gender_fig.update_layout(bargap=0.5),
    
    # 4. Monthly Sales Trend by Age Group
    filtered_df['Month'] = filtered_df['Purchase Date'].dt.to_period('M').astype(str)
    age_bins = pd.cut(filtered_df['Customer Age'], bins=[18, 25, 35, 45, 60, 80], labels=['18-25', '26-35', '36-45', '46-60', '61+'])
    age_trend_data = filtered_df.copy()
    age_trend_data['Age Group'] = age_bins
    age_trend_grouped = age_trend_data.groupby(['Month', 'Age Group'])['Total Purchase Amount'].sum().reset_index()
    age_trend_fig = px.line(age_trend_grouped, x='Month', y='Total Purchase Amount', color='Age Group',
                            title="Monthly Sales Trend by Age Group", markers=True)

    # 5. Sales Trends by Gender Over Time
    gender_trend_grouped = filtered_df.groupby([filtered_df['Purchase Date'].dt.to_period('M').astype(str), 'Gender'])['Total Purchase Amount'].sum().reset_index()
    gender_trend_fig = px.line(gender_trend_grouped, x='Purchase Date', y='Total Purchase Amount', color='Gender',
                               title="Monthly Sales Trend by Gender", markers=True)

    return ts_fig, cat_fig, gender_fig, age_trend_fig, gender_trend_fig

@app.callback(
    Output('sales-recommendation-box', 'children'),
    [Input('sales-date-range', 'start_date'),
     Input('sales-date-range', 'end_date')]
)
def update_sales_recommendations(start_date, end_date):
    if not start_date or not end_date:
        raise dash.exceptions.PreventUpdate

    filtered_df = df[(df['Purchase Date'] >= start_date) & (df['Purchase Date'] <= end_date)]

    total_sales = filtered_df['Total Purchase Amount'].sum()
    top_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().idxmax()

    recommendations = []

    if total_sales < 100000:
        recommendations.append(f"⚠️ Sales during this period are relatively low (₹{total_sales:,.0f}). Consider marketing campaigns.")
    else:
        recommendations.append(f"✅ Strong sales observed (₹{total_sales:,.0f}). Maintain consistency with top-performing channels.")

    recommendations.append(f"📈 Top-performing category: **{top_category}**. Focus on expanding inventory or bundling offers.")

    return html.Ul([html.Li(rec) for rec in recommendations])


# Tab 3: Customer Behaviour
tab3_layout = html.Div([
    html.Div([
            html.H4("Customer Insights Recommendations", style={'marginBottom': '15px', 'color': '#fbc531'}),
            html.Div(id='customer-insight-recommendation-box')
        ], style={
            'marginTop': '30px',
            'padding': '20px',
            'backgroundColor': '#2f3640',
            'color': '#f5f6fa',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.5)',
            'fontSize': '18px'
        }),
    html.Div([
        html.Div(dcc.Graph(id='age-churn-return'), style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
        html.Div(dcc.Graph(id='gender-churn-return'), style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    html.Div([
        dcc.Graph(id='correlation-heatmap')
    ], style={'marginTop': '30px'})
])

@app.callback(
    Output('customer-insight-recommendation-box', 'children'),
    Input('tabs', 'value'),
)
def generate_customer_insight_recommendations(tab):
    if tab != 'tab3':
        raise dash.exceptions.PreventUpdate

    df_filtered = df.copy()

    clv_median = df_filtered['CLV'].median()
    churn_rate = df_filtered['Churn'].mean()
    return_rate = df_filtered['Returns'].mean()

    recommendations = []

    if clv_median > 1000:
        recommendations.append("🎯 Target high-value customers for exclusive offers and personalized discounts.")
    else:
        recommendations.append("📊 Consider increasing engagement through loyalty programs for mid-tier customers.")

    if churn_rate > 0.2:
        recommendations.append("⚠️ High churn rate detected! Consider improving customer retention strategies.")
    else:
        recommendations.append("👍 Customer retention seems strong. Continue with the current engagement efforts.")

    if return_rate > 0.1:
        recommendations.append("🔄 High return rate! Investigate the reasons for returns and enhance product quality.")
    else:
        recommendations.append("✅ Product satisfaction is good. Keep up the quality control.")

    return html.Div([
        html.P(rec, style={'margin': '10px 0', 'fontSize': '16px', 'color': '#f5f6fa'}) for rec in recommendations
    ])

@app.callback(
    [Output('age-churn-return', 'figure'),
     Output('gender-churn-return', 'figure'),
     Output('correlation-heatmap', 'figure')],
    Input('tabs', 'value')
)
def update_customer_behavior(tab):
    if tab != 'tab3':
        raise dash.exceptions.PreventUpdate

    # Bin ages
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ['18–25', '26–35', '36–45', '46–55', '56–65', '65+']
    df['Age Group'] = pd.cut(df['Customer Age'], bins=age_bins, labels=age_labels, right=False)

    # Group by Age Group
    age_churn_binned = df.groupby('Age Group')[['Churn', 'Returns']].sum().reset_index()

    # Plot
    age_churn_fig = go.Figure()

    # Bars
    age_churn_fig.add_trace(go.Bar(
        x=age_churn_binned['Age Group'],
        y=age_churn_binned['Churn'],
        name='Churn',
        marker_color='#66c2a5'
    ))
    age_churn_fig.add_trace(go.Bar(
        x=age_churn_binned['Age Group'],
        y=age_churn_binned['Returns'],
        name='Returns',
        marker_color='#fc8d62'
    ))

    # Lines
    age_churn_fig.add_trace(go.Scatter(
        x=age_churn_binned['Age Group'],
        y=age_churn_binned['Churn'],
        name='Churn (Line)',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        showlegend=False
    ))
    age_churn_fig.add_trace(go.Scatter(
        x=age_churn_binned['Age Group'],
        y=age_churn_binned['Returns'],
        name='Returns (Line)',
        mode='lines+markers',
        line=dict(color='#d62728', width=2),
        showlegend=False
    ))

    # Layout
    age_churn_fig.update_layout(
        title='Age Group vs Churn and Return',
        xaxis_title='Age Group',
        yaxis_title='Count',
        barmode='group',
        plot_bgcolor='#1e272e',
        paper_bgcolor='#1e272e',
        font=dict(color='#ecf0f1'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Gender vs Churn & Return
    gender_churn = df.groupby('Gender')[['Churn', 'Returns']].sum().reset_index()
    gender_churn_fig = px.bar(gender_churn, x='Gender', y=['Churn', 'Returns'],
                              title='Gender vs Churn and Return', barmode='group',
                              color_discrete_sequence=px.colors.qualitative.Set3)

    # Heatmap of numeric correlation
    corr_cols = ['Customer Age', 'Total Purchase Amount', 'Churn', 'Returns']
    corr_matrix = df[corr_cols].corr()
    heatmap_fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Tealrose',
                            title="Correlation Heatmap: Age, Purchase, Churn, Return")

    return age_churn_fig, gender_churn_fig, heatmap_fig

# Tab 4: Product Performance
tab4_layout = html.Div([
    html.Div([
            html.H4("Product Performance Recommendations", style={'marginBottom': '15px', 'color': '#fbc531'}),
            html.Div(id='product-performance-recommendation-box')
        ], style={
            'marginTop': '30px',
            'padding': '20px',
            'backgroundColor': '#2f3640',
            'color': '#f5f6fa',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.5)',
            'fontSize': '18px'
        }),
    html.Div([
        dcc.Graph(id='category-demand', style={'width': '32%', 'display': 'inline-block'}),
        dcc.Graph(id='category-revenue', style={'width': '32%', 'display': 'inline-block', 'marginLeft': '2%'}),
        dcc.Graph(id='category-return-rate', style={'width': '32%', 'display': 'inline-block', 'marginLeft': '2%'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'})
])


@app.callback(
    [Output('category-demand', 'figure'),
     Output('category-revenue', 'figure'),
     Output('category-return-rate', 'figure')],
    Input('tabs', 'value')
)
def update_product_performance(tab):
    if tab != 'tab4':
        raise dash.exceptions.PreventUpdate

    # Product Category Demand (Count)
    demand = df['Product Category'].value_counts().reset_index()
    demand.columns = ['Product Category', 'Order Count']
    demand_fig = px.bar(demand, x='Product Category', y='Order Count',
                        title="Product Category Demand", color='Product Category',
                        color_discrete_sequence=px.colors.sequential.Blues)

    # Product Category Revenue (Total Purchase)
    revenue = df.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
    revenue_fig = px.bar(revenue, x='Product Category', y='Total Purchase Amount',
                         title="Product Category Revenue", color='Product Category',
                         color_discrete_sequence=px.colors.sequential.Greens)

    # Product Category Return Rate
    returns = df.groupby('Product Category').agg({
        'Returns': 'sum',
        'Customer ID': 'count'
    }).reset_index()
    returns['Return Rate (%)'] = (returns['Returns'] / returns['Customer ID']) * 100
    return_fig = px.bar(returns, x='Product Category', y='Return Rate (%)',
                        title="Product Category Return Rate", color='Product Category',
                        color_discrete_sequence=px.colors.sequential.Oranges)

    return demand_fig, revenue_fig, return_fig

@app.callback(
    Output('product-performance-recommendation-box', 'children'),
    Input('tabs', 'value')
)
def generate_product_performance_recommendations(tab):
    if tab != 'tab4':
        raise dash.exceptions.PreventUpdate

    top_categories = df['Product Category'].value_counts().head(3).index.tolist()
    high_return_category = df[df['Returns'] == 1]['Product Category'].value_counts().idxmax()

    recs = [
        f"📦 Focus on top-selling categories like {', '.join(top_categories)} to optimize stock planning.",
        f"🔄 Category with the highest return rate: **{high_return_category}** – consider reviewing product quality or description.",
        "📈 Increase promotions on categories with high revenue but low purchase frequency."
    ]

    return html.Div([
        html.P(r, style={'margin': '10px 0', 'fontSize': '16px'}) for r in recs
    ])


# Tab 5 : Market Basket Analysis
tab5_layout = html.Div([
    html.Div([
        html.H4("Market Basket Recommendations", style={'marginBottom': '15px', 'color': '#fbc531'}),
        html.Div(id='market-basket-recommendation-box')
    ], style={
        'marginTop': '30px',
        'padding': '20px',
        'backgroundColor': '#2f3640',
        'color': '#f5f6fa',
        'borderRadius': '10px',
        'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.5)',
        'fontSize': '18px'
    }),
    dcc.Graph(id='frequent-itemsets-bar', style={'marginTop': '30px'}),
])

@app.callback(
    Output('frequent-itemsets-bar', 'figure'),
    Input('tabs', 'value')
)
def update_frequent_itemsets(tab):
    if tab != 'tab5':
        raise dash.exceptions.PreventUpdate

    # Top 10 itemsets
    top_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).head(10)
    top_itemsets['itemsets_str'] = top_itemsets['itemsets'].apply(
        lambda x: re.sub(r"frozenset\(\{?(.*?)\}?\)", r"\1", str(x)).replace("'", "")
    )

    # Create plot
    fig = px.bar(top_itemsets, x='itemsets_str', y='support',
                 title='Top 10 Frequent Itemsets',
                 color='support',
                 color_continuous_scale='teal')

    fig.update_layout(xaxis_title="Itemsets", yaxis_title="Support")

    return fig


@app.callback(
    Output('market-basket-recommendation-box', 'children'),
    Input('tabs', 'value')
)
def generate_market_basket_recommendations(tab):
    if tab != 'tab5':
        raise dash.exceptions.PreventUpdate

    # Make sure 'rules' exists and has data
    if rules.empty:
        return html.P("📈 Use bundling strategies to promote sales in frequently purchased itemsets")

    def format_items(itemset):
        return ', '.join(sorted(list(itemset)))

    top_rules = rules.sort_values(by='lift', ascending=False).head(3)

    recommendations = []
    for _, rule in top_rules.iterrows():
        antecedents = format_items(rule['antecedents'])
        consequents = format_items(rule['consequents'])
        lift = rule['lift']
        recommendations.append(
            html.P(f"🛒 Customers who buy **{antecedents}** also tend to buy **{consequents}** (Lift: {lift:.2f})")
        )

    return recommendations

# 6. Run App

server = app.server #Required by Render

if __name__ == '__main__':
    app.run(debug=False)
