# 🛡️ Sentinel Financial AI - User Guide

## 📋 How to Use All Advanced Features

This guide explains exactly how to use each of the three advanced features in your Sentinel Financial AI system.

## 🚀 Quick Start

1. **Upload Documents** (if not already done)
   - Go to sidebar → "📤 Upload New Document"
   - Upload PDF financial reports (10-K, 10-Q, etc.)
   - Click "➕ Process & Index" to prepare documents

2. **Activate Documents**
   - Select documents in sidebar → "📋 Select Reports to Analyze"
   - Click "🚀 Activate Selection"

3. **Use Advanced Features**
   - Navigate to features using the buttons at the top

---

## 🛡️ Security Dashboard

### **What it does:**
Measures AI truthfulness and accuracy using RAGAS framework

### **How to use:**
1. **Ask questions first** in the main chat interface
2. **Navigate to Security Dashboard** using the "🛡️ Security Dashboard" button
3. **View metrics** automatically generated from your queries

### **Key Metrics:**
- **Faithfulness**: How often AI answers are supported by retrieved context
- **Answer Relevance**: How well answers address your questions
- **Context Precision**: How relevant retrieved context chunks are
- **Context Recall**: How much relevant information was retrieved

### **Example workflow:**
1. Ask: "What was Tesla's revenue in 2023?"
2. Ask: "Compare Apple and Tesla's profit margins"
3. Go to Security Dashboard to see evaluation metrics

---

## 🕸️ Knowledge Graph

### **What it does:**
Visualizes relationships between entities (companies, people, locations, financial metrics)

### **How to use:**
1. **Activate documents** in sidebar (required step)
2. **Navigate to Knowledge Graph** using "🕸️ Knowledge Graph" button
3. **Select collections** for graph building
4. **Click "🔄 Build Knowledge Graph"**
5. **Wait for processing** (extracts entities and relationships)
6. **View interactive graph** with clickable nodes

### **What you'll see:**
- 🔴 **Companies**: Apple Inc., Tesla Motors, etc.
- 🔵 **People**: Executives, key personnel
- 🟢 **Locations**: Headquarters, facilities, offices
- 🟡 **Financial Metrics**: Revenue, profit, expenses
- **Lines connecting related entities**

### **Example workflow:**
1. Activate "apple10K" and "tesla10k" documents
2. Click "Build Knowledge Graph"
3. Explore the interactive visualization
4. Click nodes to see detailed information

---

## 🤖 Multi-Step Agent

### **What it does:**
Performs complex multi-step analytical workflows using specialized tools

### **How to use:**
1. **Activate documents** in sidebar (required step)
2. **Navigate to Multi-Step Agent** using "🤖 Multi-Step Agent" button
3. **Enter complex questions** that require multiple steps
4. **Watch the agent work** through the analysis
5. **Review the comprehensive report**

### **Available Tools:**
- **Financial Search**: Multi-document search capabilities
- **Calculator**: Financial calculations (growth rates, percentages, ratios)
- **Summarizer**: Text condensation and key insights extraction
- **Comparison Tool**: Multi-document data comparison

### **Example questions:**
- "Compare Tesla's and Apple's revenue growth over the past year"
- "Analyze the risk factors affecting both companies"
- "Calculate and compare profit margins between the reports"
- "What are the key differences in their cash flow statements?"

### **Example workflow:**
1. Activate both Apple and Tesla documents
2. Ask: "Compare the revenue growth and profit margins of Apple and Tesla"
3. Watch the agent plan and execute the analysis
4. Review the detailed multi-step report

---

## 📊 Feature Comparison

| Feature | When to Use | What You Get | Time Required |
|---------|-------------|--------------|---------------|
| **Security Dashboard** | After asking questions | Quality metrics and performance tracking | Instant |
| **Knowledge Graph** | To understand relationships | Visual entity relationship map | 1-2 minutes |
| **Multi-Step Agent** | For complex analysis | Comprehensive multi-step reports | 30 seconds - 2 minutes |

---

## 💡 Pro Tips

### **For Best Results:**

1. **Upload multiple documents** for richer analysis
2. **Ask specific questions** for better agent performance
3. **Use the Security Dashboard** to monitor AI quality
4. **Build knowledge graphs** to discover hidden relationships
5. **Try complex multi-step questions** for the agent

### **Common Workflows:**

**Workflow 1: Quality Assurance**
1. Ask questions in main chat
2. Check Security Dashboard for metrics
3. Improve question quality based on feedback

**Workflow 2: Relationship Discovery**
1. Activate multiple documents
2. Build knowledge graph
3. Explore entity relationships
4. Export graph for further analysis

**Workflow 3: Complex Analysis**
1. Activate relevant documents
2. Ask multi-step questions to agent
3. Review comprehensive reports
4. Use insights for decision making

---

## 🔧 Troubleshooting

### **Security Dashboard shows "No evaluations available":**
- ✅ **Solution**: Ask questions in the main chat first, then check the dashboard

### **Knowledge Graph shows "No graph available":**
- ✅ **Solution**: 1) Activate documents in sidebar, 2) Click "Build Knowledge Graph"

### **Multi-Step Agent not working:**
- ✅ **Solution**: Ensure documents are activated before using the agent

### **Slow performance:**
- ✅ **Solution**: Process fewer documents at once or use smaller files

---

## 🎯 Ready to Start?

Your Sentinel Financial AI system is now ready with all advanced features! 

**Start here:**
1. Upload and activate documents
2. Ask some questions in the main chat
3. Explore the advanced features
4. Share your impressive AI capabilities on LinkedIn! 🚀

---

## 📞 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all documents are properly activated
3. Verify the Streamlit application is running
4. Check the console for any error messages

**Application URL:** http://localhost:8503

**Ready for LinkedIn showcase!** 🎉