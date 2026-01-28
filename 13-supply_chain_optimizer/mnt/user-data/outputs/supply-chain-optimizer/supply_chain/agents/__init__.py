"""
Supply Chain Optimizer - CrewAI Agents

Hierarchical structure:
- Strategic: Supply Chain Director
- Tactical: Demand, Inventory, and Logistics Managers
- Operational: Specialized agents for forecasting, replenishment, routing, etc.
"""

from typing import Optional, Any
import os
from datetime import datetime

try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    
    # Mock classes for when CrewAI is not available
    class Agent:
        def __init__(self, **kwargs):
            self.role = kwargs.get('role', '')
            self.goal = kwargs.get('goal', '')
            self.backstory = kwargs.get('backstory', '')
            self.verbose = kwargs.get('verbose', False)
            self.allow_delegation = kwargs.get('allow_delegation', False)
            self.tools = kwargs.get('tools', [])
    
    class Task:
        def __init__(self, **kwargs):
            self.description = kwargs.get('description', '')
            self.expected_output = kwargs.get('expected_output', '')
            self.agent = kwargs.get('agent', None)
            self.context = kwargs.get('context', [])
    
    class Crew:
        def __init__(self, **kwargs):
            self.agents = kwargs.get('agents', [])
            self.tasks = kwargs.get('tasks', [])
            self.process = kwargs.get('process', None)
            self.verbose = kwargs.get('verbose', False)
        
        def kickoff(self, inputs=None):
            return "CrewAI not available - running in simulation mode"
    
    class Process:
        sequential = "sequential"
        hierarchical = "hierarchical"


# =============================================================================
# Operational Level Agents (Level 3)
# =============================================================================

class ForecastAgent:
    """Agent for demand forecasting."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Demand Forecasting Specialist",
            goal="Generate accurate demand forecasts using statistical methods and market intelligence",
            backstory="""You are an expert demand forecaster with 15 years of experience 
            in supply chain analytics. You specialize in time series analysis, seasonal 
            decomposition, and machine learning-based forecasting. You understand the 
            importance of forecast accuracy for inventory optimization and have developed 
            proprietary methods for improving predictions in volatile markets.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
        )
    
    @staticmethod
    def create_forecast_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Analyze historical demand data and generate forecasts for the next quarter.
            
            Your analysis should include:
            1. Review 12 months of historical demand by SKU
            2. Identify trends (increasing, decreasing, stable)
            3. Detect seasonality patterns
            4. Calculate forecast using appropriate method (moving average, exponential smoothing)
            5. Provide confidence intervals (95% level)
            6. Flag any anomalies or data quality issues
            
            Focus on high-value (ABC class A) items first.""",
            expected_output="""A demand forecast report containing:
            - Forecast quantities by SKU for next 3 months
            - Confidence intervals (lower/upper bounds)
            - Trend analysis summary
            - Seasonality patterns identified
            - Forecast method used for each SKU
            - Data quality notes""",
            agent=agent,
            context=context or [],
        )


class MarketAnalystAgent:
    """Agent for market intelligence analysis."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Market Intelligence Analyst",
            goal="Analyze market trends, competitor activities, and external factors affecting demand",
            backstory="""You are a market research expert who monitors industry trends, 
            economic indicators, and competitive dynamics. You track factors like commodity 
            prices, trade policies, and consumer sentiment that impact supply chain 
            planning. Your insights help adjust forecasts and identify risks early.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
        )
    
    @staticmethod
    def create_analysis_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Analyze external market factors that could impact demand and supply.
            
            Your analysis should cover:
            1. Industry growth trends and projections
            2. Raw material price trends (steel, copper, electronics)
            3. Trade policy impacts (tariffs, regulations)
            4. Competitor activity and market share shifts
            5. Economic indicators affecting demand
            6. Technology trends affecting product lifecycle
            
            Provide adjustment recommendations for base forecasts.""",
            expected_output="""Market intelligence report with:
            - Key market trends summary
            - Demand adjustment factors by product category
            - Risk factors identified
            - Opportunity areas
            - Recommended forecast adjustments (+/- %)
            - Confidence level of market predictions""",
            agent=agent,
            context=context or [],
        )


class ReplenishmentAgent:
    """Agent for inventory replenishment recommendations."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Inventory Replenishment Specialist",
            goal="Optimize replenishment orders to maintain service levels while minimizing costs",
            backstory="""You are an inventory optimization expert who calculates optimal 
            order quantities, timing, and supplier allocation. You understand EOQ theory, 
            MRP logic, and just-in-time principles. You balance the competing goals of 
            high service levels, low inventory costs, and supply chain efficiency.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
        )
    
    @staticmethod
    def create_replenishment_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Generate replenishment recommendations based on inventory levels and forecasts.
            
            For each SKU:
            1. Review current inventory position (on-hand, on-order, in-transit)
            2. Calculate projected inventory over forecast horizon
            3. Determine if reorder is needed (compare to reorder point)
            4. Calculate Economic Order Quantity (EOQ)
            5. Consider minimum order quantities and multiples
            6. Recommend order quantity and timing
            7. Suggest supplier allocation
            
            Prioritize items with stockout risk or critical status.""",
            expected_output="""Replenishment plan containing:
            - Order recommendations by SKU
            - Suggested order quantities
            - Recommended suppliers
            - Order timing (dates)
            - Priority level (Critical/High/Normal/Low)
            - Estimated costs
            - Service level impact analysis""",
            agent=agent,
            context=context or [],
        )


class SafetyStockAgent:
    """Agent for safety stock optimization."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Safety Stock Optimization Specialist",
            goal="Optimize safety stock levels to balance service levels against inventory costs",
            backstory="""You are an expert in statistical inventory management with deep 
            knowledge of service level theory, demand variability, and lead time uncertainty. 
            You optimize safety stock to meet target service levels while minimizing 
            excess inventory and carrying costs.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
        )
    
    @staticmethod
    def create_optimization_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Optimize safety stock levels for all products.
            
            For each SKU:
            1. Analyze demand variability (standard deviation)
            2. Assess lead time variability
            3. Calculate safety stock for target service level
            4. Compare current vs. recommended safety stock
            5. Estimate impact on working capital
            6. Consider ABC classification in service level targets
            
            A-class items: 98% service level
            B-class items: 95% service level
            C-class items: 90% service level""",
            expected_output="""Safety stock optimization report:
            - Current vs. recommended safety stock by SKU
            - Service level targets by ABC class
            - Working capital impact (increase/decrease)
            - Stockout risk analysis
            - Implementation priorities""",
            agent=agent,
            context=context or [],
        )


class RouteOptimizerAgent:
    """Agent for logistics route optimization."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Logistics Route Optimization Specialist",
            goal="Optimize shipping routes and consolidate shipments to reduce costs and emissions",
            backstory="""You are a logistics optimization expert specializing in route 
            planning, load optimization, and network design. You use algorithms and 
            heuristics to minimize transportation costs while meeting delivery requirements. 
            You also consider sustainability and carbon footprint reduction.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
        )
    
    @staticmethod
    def create_optimization_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Optimize shipping routes and consolidate shipments.
            
            Analysis should include:
            1. Review pending shipments and destinations
            2. Identify consolidation opportunities
            3. Optimize delivery routes (minimize distance/time)
            4. Calculate cost savings from consolidation
            5. Estimate carbon footprint reduction
            6. Consider delivery time windows and constraints
            
            Balance cost optimization with service requirements.""",
            expected_output="""Route optimization report:
            - Consolidated shipment groups
            - Optimized route sequences
            - Before/after comparison (routes, cost, distance)
            - Carbon footprint impact
            - Delivery schedule
            - Recommendations for network improvements""",
            agent=agent,
            context=context or [],
        )


class CarrierSelectorAgent:
    """Agent for carrier selection and rate comparison."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Carrier Selection and Procurement Specialist",
            goal="Select optimal carriers based on cost, service, and reliability",
            backstory="""You are a freight procurement expert who manages carrier 
            relationships and negotiates rates. You evaluate carriers on multiple 
            criteria including cost, on-time performance, damage rates, and service 
            quality. You optimize the carrier mix to balance cost and service.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
        )
    
    @staticmethod
    def create_selection_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Select optimal carriers for pending shipments.
            
            For each shipment:
            1. Gather weight, volume, origin, destination, delivery requirements
            2. Query available carrier rates
            3. Evaluate carriers on cost, transit time, reliability
            4. Consider carrier capacity and availability
            5. Recommend primary and backup carriers
            6. Calculate total landed cost
            
            Consider both domestic and international requirements.""",
            expected_output="""Carrier selection report:
            - Recommended carriers by shipment/route
            - Rate comparison matrix
            - Service level comparison
            - Total cost analysis
            - Backup options
            - Carrier performance scorecard""",
            agent=agent,
            context=context or [],
        )


# =============================================================================
# Tactical Level Agents (Level 2) - Managers
# =============================================================================

class DemandPlanningManager:
    """Manager for the demand planning team."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Demand Planning Manager",
            goal="Coordinate demand forecasting and market analysis to provide accurate demand plans",
            backstory="""You are a senior demand planning manager with 20 years of 
            experience in S&OP (Sales & Operations Planning). You coordinate between 
            forecasters, market analysts, and sales teams to develop consensus demand 
            plans. You understand how to integrate statistical forecasts with market 
            intelligence and sales input.""",
            verbose=True,
            allow_delegation=True,
            tools=[],
        )
    
    @staticmethod
    def create_planning_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Coordinate demand planning activities and produce consensus demand plan.
            
            Responsibilities:
            1. Review statistical forecasts from Forecast Agent
            2. Incorporate market intelligence from Market Analyst
            3. Adjust for known events (promotions, launches, discontinuations)
            4. Develop consensus demand plan
            5. Calculate forecast accuracy metrics
            6. Identify demand risks and opportunities
            
            Produce a demand plan for the next quarter by product category.""",
            expected_output="""Consensus demand plan containing:
            - Forecast by SKU and month
            - Adjustments applied (reason and amount)
            - Forecast accuracy assessment
            - Risk/opportunity summary
            - Key assumptions
            - Recommended actions""",
            agent=agent,
            context=context or [],
        )


class InventoryManager:
    """Manager for inventory management team."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Inventory Management Manager",
            goal="Optimize inventory levels to maximize service while minimizing working capital",
            backstory="""You are a supply chain director with expertise in inventory 
            strategy and optimization. You balance the trade-offs between service levels, 
            inventory costs, and working capital. You set inventory policies, manage 
            safety stock strategies, and drive continuous improvement in inventory 
            performance metrics.""",
            verbose=True,
            allow_delegation=True,
            tools=[],
        )
    
    @staticmethod
    def create_management_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Coordinate inventory optimization and generate replenishment plan.
            
            Responsibilities:
            1. Review demand plan from Demand Planning
            2. Coordinate safety stock optimization
            3. Review replenishment recommendations
            4. Validate supplier selection
            5. Approve purchase orders
            6. Monitor inventory KPIs
            
            Produce an inventory action plan with specific recommendations.""",
            expected_output="""Inventory management report:
            - Inventory status summary (by category, ABC class)
            - Replenishment plan (orders to place)
            - Safety stock adjustments
            - Supplier allocation
            - Working capital impact
            - Service level forecast
            - Risk items requiring attention""",
            agent=agent,
            context=context or [],
        )


class LogisticsManager:
    """Manager for logistics planning team."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Logistics Planning Manager",
            goal="Optimize logistics operations to reduce costs while maintaining delivery performance",
            backstory="""You are a logistics director responsible for transportation 
            strategy, carrier management, and distribution network optimization. You 
            manage relationships with carriers, optimize freight spend, and drive 
            sustainability initiatives. You balance cost efficiency with service 
            excellence.""",
            verbose=True,
            allow_delegation=True,
            tools=[],
        )
    
    @staticmethod
    def create_planning_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Coordinate logistics planning and optimization.
            
            Responsibilities:
            1. Review inbound shipments from purchase orders
            2. Plan outbound distribution
            3. Coordinate route optimization
            4. Approve carrier selections
            5. Monitor logistics KPIs
            6. Identify cost reduction opportunities
            
            Produce logistics plan with cost and sustainability analysis.""",
            expected_output="""Logistics management report:
            - Shipment plan (inbound and outbound)
            - Carrier assignments
            - Route optimization summary
            - Cost analysis (current vs. optimized)
            - Sustainability metrics (carbon footprint)
            - Performance KPIs
            - Improvement recommendations""",
            agent=agent,
            context=context or [],
        )


# =============================================================================
# Strategic Level Agent (Level 1) - Director
# =============================================================================

class SupplyChainDirector:
    """Strategic director overseeing the entire supply chain."""
    
    @staticmethod
    def create() -> Agent:
        return Agent(
            role="Supply Chain Director",
            goal="Optimize end-to-end supply chain performance through strategic decisions and KPI monitoring",
            backstory="""You are the Chief Supply Chain Officer with 25 years of 
            experience across multiple industries. You take a holistic view of the 
            supply chain, balancing customer service, cost efficiency, working capital, 
            and risk management. You make strategic decisions on network design, 
            sourcing strategy, and technology investments. You report to the C-suite 
            and translate business strategy into supply chain execution.""",
            verbose=True,
            allow_delegation=True,
            tools=[],
        )
    
    @staticmethod
    def create_strategic_task(agent: Agent, context: list = None) -> Task:
        return Task(
            description="""Provide strategic oversight and produce executive supply chain report.
            
            Strategic responsibilities:
            1. Review demand plan and assess business alignment
            2. Evaluate inventory strategy and working capital impact
            3. Assess logistics efficiency and cost position
            4. Monitor supply chain risks
            5. Identify strategic improvement opportunities
            6. Make resource allocation decisions
            
            Produce executive summary for leadership review.""",
            expected_output="""Executive Supply Chain Report:
            
            1. EXECUTIVE SUMMARY
            - Key metrics dashboard
            - Critical issues requiring attention
            - Strategic recommendations
            
            2. DEMAND OUTLOOK
            - Forecast summary and confidence
            - Market factors and risks
            
            3. INVENTORY POSITION
            - Value and turnover
            - Service level performance
            - Working capital status
            
            4. LOGISTICS PERFORMANCE
            - Cost efficiency
            - Carrier performance
            - Sustainability metrics
            
            5. RISK ASSESSMENT
            - Supply risks
            - Demand risks
            - Logistics risks
            
            6. ACTION ITEMS
            - Prioritized recommendations
            - Resource requirements
            - Timeline""",
            agent=agent,
            context=context or [],
        )


# =============================================================================
# Crew Assembly
# =============================================================================

def create_supply_chain_crew(verbose: bool = True) -> Crew:
    """
    Create the full hierarchical supply chain crew.
    
    Structure:
    - Strategic: Supply Chain Director
    - Tactical: Demand Manager, Inventory Manager, Logistics Manager
    - Operational: 6 specialized agents
    """
    
    # Create operational agents
    forecast_agent = ForecastAgent.create()
    market_analyst = MarketAnalystAgent.create()
    replenishment_agent = ReplenishmentAgent.create()
    safety_stock_agent = SafetyStockAgent.create()
    route_optimizer = RouteOptimizerAgent.create()
    carrier_selector = CarrierSelectorAgent.create()
    
    # Create tactical managers
    demand_manager = DemandPlanningManager.create()
    inventory_manager = InventoryManager.create()
    logistics_manager = LogisticsManager.create()
    
    # Create strategic director
    director = SupplyChainDirector.create()
    
    # Create operational tasks
    forecast_task = ForecastAgent.create_forecast_task(forecast_agent)
    market_task = MarketAnalystAgent.create_analysis_task(market_analyst)
    replenishment_task = ReplenishmentAgent.create_replenishment_task(replenishment_agent)
    safety_stock_task = SafetyStockAgent.create_optimization_task(safety_stock_agent)
    route_task = RouteOptimizerAgent.create_optimization_task(route_optimizer)
    carrier_task = CarrierSelectorAgent.create_selection_task(carrier_selector)
    
    # Create tactical tasks (depend on operational tasks)
    demand_planning_task = DemandPlanningManager.create_planning_task(
        demand_manager, 
        context=[forecast_task, market_task]
    )
    inventory_task = InventoryManager.create_management_task(
        inventory_manager,
        context=[demand_planning_task, replenishment_task, safety_stock_task]
    )
    logistics_task = LogisticsManager.create_planning_task(
        logistics_manager,
        context=[route_task, carrier_task]
    )
    
    # Create strategic task (depends on tactical tasks)
    strategic_task = SupplyChainDirector.create_strategic_task(
        director,
        context=[demand_planning_task, inventory_task, logistics_task]
    )
    
    # Assemble crew with hierarchical process
    crew = Crew(
        agents=[
            # Strategic
            director,
            # Tactical
            demand_manager, inventory_manager, logistics_manager,
            # Operational
            forecast_agent, market_analyst, replenishment_agent,
            safety_stock_agent, route_optimizer, carrier_selector,
        ],
        tasks=[
            # Operational tasks first
            forecast_task, market_task,
            replenishment_task, safety_stock_task,
            route_task, carrier_task,
            # Then tactical tasks
            demand_planning_task, inventory_task, logistics_task,
            # Finally strategic task
            strategic_task,
        ],
        process=Process.hierarchical if CREWAI_AVAILABLE else "hierarchical",
        verbose=verbose,
    )
    
    return crew


def get_agent_hierarchy() -> dict:
    """Get the agent hierarchy structure."""
    return {
        "strategic": {
            "Supply Chain Director": {
                "role": "Strategic decisions, KPI monitoring",
                "reports_to": "C-Suite",
            }
        },
        "tactical": {
            "Demand Planning Manager": {
                "role": "Demand forecasting coordination",
                "reports_to": "Supply Chain Director",
                "team": ["Forecast Agent", "Market Analyst"],
            },
            "Inventory Management Manager": {
                "role": "Inventory optimization",
                "reports_to": "Supply Chain Director", 
                "team": ["Replenishment Agent", "Safety Stock Agent"],
            },
            "Logistics Planning Manager": {
                "role": "Logistics and transportation",
                "reports_to": "Supply Chain Director",
                "team": ["Route Optimizer", "Carrier Selector"],
            },
        },
        "operational": {
            "Forecast Agent": "Demand forecasting (ML-based)",
            "Market Analyst": "Market intelligence analysis",
            "Replenishment Agent": "Order quantity optimization",
            "Safety Stock Agent": "Safety stock calculation",
            "Route Optimizer": "Route optimization",
            "Carrier Selector": "Carrier selection and rates",
        },
    }
